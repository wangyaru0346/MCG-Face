import torch
import torch.optim as optim
import os
import sys
import numpy as np
import configparser
import trimesh
import csv
from scipy.spatial import cKDTree

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
sys.path.append(project_root)

from funcs import load_generator

CORRECT_CHECKPOINT_DIR = "checkpoints"
MASK_input_DIR = "examples" 
OUTPUT_DIR_NAME = "results_ablation"

SPARSITY_RATIO = 0.10 

def save_masked_mesh(verts, faces, keep_mask, path):
    if torch.is_tensor(verts): verts = verts.detach().cpu().numpy()
    if torch.is_tensor(faces): faces = faces.detach().cpu().numpy()
    if torch.is_tensor(keep_mask): keep_mask = keep_mask.detach().cpu().numpy()
    valid_faces_idx = keep_mask[faces[:, 0]] & keep_mask[faces[:, 1]] & keep_mask[faces[:, 2]]
    new_faces = faces[valid_faces_idx]
    trimesh.Trimesh(vertices=verts, faces=new_faces, process=False).export(path)

def compute_mean_error_mm_local(pred_verts, gt_verts, hole_mask):
    dist = torch.norm(pred_verts - gt_verts, dim=1)
    valid_dist = dist[hole_mask]
    if valid_dist.numel() == 0: return 0.0
    return valid_dist.mean().item()

def compute_advanced_metrics(pred_verts, gt_verts, pred_normals, gt_normals, hole_mask):
    p_pred = pred_verts[hole_mask]
    p_gt = gt_verts[hole_mask]
    n_pred = pred_normals[hole_mask]
    n_gt = gt_normals[hole_mask]
    if p_pred.shape[0] == 0: return 0, 0
    
    dist_sq = torch.sum((p_pred - p_gt)**2, dim=1)
    hd = torch.sqrt(dist_sq).max().item() * 1000 
    
    dots = torch.sum(n_pred * n_gt, dim=1)
    dots = torch.clamp(dots, -1.0, 1.0)
    rad_errors = torch.acos(dots)
    normal_error = torch.rad2deg(rad_errors).mean().item()
    return hd, normal_error

def compute_L_fit(current_verts, target_verts, weights):
    return (torch.sum((current_verts - target_verts)**2, dim=1) * weights).mean()

def compute_L_local(current_verts, target_verts, edge_index, hole_mask):
    src, dst = edge_index[0], edge_index[1]
    curr_edges = current_verts[src] - current_verts[dst]
    target_edges = target_verts[src] - target_verts[dst]
    mask_edges = hole_mask[src] | hole_mask[dst]
    if mask_edges.sum() == 0: return torch.tensor(0.0, device=current_verts.device)
    return torch.mean((curr_edges[mask_edges] - target_edges[mask_edges]) ** 2)

def project_points(verts): return verts[:, :2]

def compute_L_global(current_verts, target_verts, edge_index, hole_mask):
    curr_2d = project_points(current_verts)
    target_2d = project_points(target_verts)
    src, dst = edge_index[0], edge_index[1]
    curr_edges = curr_2d[src] - curr_2d[dst]
    target_edges = target_2d[src] - target_2d[dst]
    mask_edges = hole_mask[src] | hole_mask[dst]
    if mask_edges.sum() == 0: return torch.tensor(0.0, device=current_verts.device)
    return torch.mean((curr_edges[mask_edges] - target_edges[mask_edges]) ** 2)

def compute_L_reg(delta): return torch.mean(delta**2)

def get_edges_from_faces(faces_np, device):
    edges = set()
    for f in faces_np:
        edges.add(tuple(sorted((f[0], f[1])))); edges.add(tuple(sorted((f[1], f[2])))); edges.add(tuple(sorted((f[2], f[0]))) )
    directed = []; 
    for u, v in edges: directed.append([u, v]); directed.append([v, u])
    return torch.tensor(directed, dtype=torch.long).t().contiguous().to(device)

def get_symmetry_indices(verts_np):
    flipped_verts = verts_np.copy(); flipped_verts[:, 0] = -flipped_verts[:, 0]
    tree = cKDTree(verts_np); dists, indices = tree.query(flipped_verts, k=1)
    return torch.from_numpy(indices).long()

def get_boundary_mask(n_verts, keep_mask, edge_index):
    boundary = torch.zeros(n_verts, dtype=torch.bool, device=keep_mask.device)
    src, dst = edge_index[0], edge_index[1]
    is_boundary = keep_mask[src] & (~keep_mask[dst])
    boundary[src[is_boundary]] = True
    return boundary

def normalize_to_unit_sphere(verts):
    if isinstance(verts, np.ndarray): verts = torch.from_numpy(verts).float()
    center = verts.mean(dim=0, keepdim=True); verts_c = verts - center
    scale = verts_c.abs().max(); 
    if scale < 1e-6: scale = torch.tensor(1.0)
    return verts_c / scale, center, scale

def create_half_face_hole(verts_tensor):
    n_verts = verts_tensor.shape[0]; device = verts_tensor.device
    keep_mask = torch.ones(n_verts, dtype=torch.bool, device=device)
    attack_center = torch.tensor([-0.2, -0.1, verts_tensor[:,2].mean()], device=device)
    keep_mask &= torch.norm(verts_tensor - attack_center, dim=1) > 0.55
    return keep_mask

class LocalConfig:
    def __init__(self, config_path):
        self.conf = {}; cp = configparser.ConfigParser(); cp.read(config_path)
        self.conf['z_length'] = cp.getint('model parameters', 'z_length')
        self.conf['batch_norm'] = cp.getboolean('model parameters', 'batch_norm')
        self.conf['num_features_global'] = [int(x) for x in cp.get('model parameters', 'num_features_global').split(',')]
        self.conf['num_features_local'] = [int(x) for x in cp.get('model parameters', 'num_features_local').split(',')]
        self.conf['down_sampling_factors'] = [float(x) for x in cp.get('model parameters', 'down_sampling_factors').split(',')]
        self.conf['n_layers'] = len(self.conf['num_features_global']) - 1
        self.conf['template_file'] = os.path.join(project_root, cp.get('I/O parameters', 'template_file'))
        self.conf['checkpoint_dir'] = CORRECT_CHECKPOINT_DIR
    def __getitem__(self, key): return self.conf[key]


ABLATION_MODES = {
    "full": {
        "weights": {"w_fit": 1.0, "w_local": 500.0, "w_global": 300.0, "w_reg": 1.0},
        "lr": 0.005,
        "boundary_w": 5000.0
    },

    "No_fit": {
        "weights": {"w_fit": 0.0, "w_local": 500.0, "w_global": 300.0, "w_reg": 1.0},
        "lr": 0.005,
        "boundary_w": 5000.0
    },

    "No_local": {
        "weights": {"w_fit": 1.0, "w_local": 0.0, "w_global": 300.0, "w_reg": 1.0},
        "lr": 0.005,
        "boundary_w": 5000.0
    },

    "No_global": {
        "weights": {"w_fit": 0.5, "w_local": 500.0, "w_global": 0.0, "w_reg": 1.0},
        "lr": 0.005,
        "boundary_w": 0.1
    },

    "No_reg": {
        "weights": {"w_fit": 1.0, "w_local": 500.0, "w_global": 300.0, "w_reg": 0.0},
        "lr": 0.2,
        "boundary_w": 5000.0
    },
}

def run_optimization_stage2(M_start, M_gt, M_target_full, edge_index, hole_mask, boundary_mask, 
                           mode_settings, sparsity_ratio):
    
    weights = mode_settings["weights"]
    learning_rate = mode_settings["lr"]
    boundary_weight = mode_settings["boundary_w"]

    device = M_start.device
    M_curr = M_start.detach().clone().requires_grad_(True)
    delta = torch.zeros_like(M_curr, requires_grad=True, device=device)
    optimizer = optim.Adam([delta], lr=learning_rate)

    weight_map_fit = torch.ones_like(hole_mask, dtype=torch.float32, device=device)
    
    weight_map_fit[~hole_mask] = 5000.0 
    
    weight_map_fit[boundary_mask] = boundary_weight
    
    torch.manual_seed(999) 
    random_mask = torch.rand_like(weight_map_fit) < sparsity_ratio
    weight_map_fit[hole_mask & (~random_mask)] = 0.0
    weight_map_fit[hole_mask & random_mask] = 5000.0

    for step in range(300): 
        optimizer.zero_grad()
        M_optim = M_start + delta
        
        loss_fit = compute_L_fit(M_optim, M_gt, weight_map_fit) if weights['w_fit'] > 0 else torch.tensor(0.0, device=device)
        loss_local = compute_L_local(M_optim, M_target_full, edge_index, hole_mask) if weights['w_local'] > 0 else torch.tensor(0.0, device=device)
        loss_global = compute_L_global(M_optim, M_target_full, edge_index, hole_mask) if weights['w_global'] > 0 else torch.tensor(0.0, device=device)
        loss_reg = compute_L_reg(delta) if weights['w_reg'] > 0 else torch.tensor(0.0, device=device)

        loss = (weights['w_fit'] * loss_fit + 
                weights['w_local'] * loss_local + 
                weights['w_global'] * loss_global + 
                weights['w_reg'] * loss_reg)
        loss.backward()
        optimizer.step()
    
    return M_start + delta

def run_ablation_study():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Info: Starting ablation study on {device}...")

    cfg_path = os.path.join(project_root, 'config/config.cfg')
    config = LocalConfig(cfg_path)
    decoder = load_generator(config).to(device).eval()
    
    norm = torch.load(os.path.join(project_root, 'checkpoints/norm.pt'), map_location=device)
    mean, std = norm['mean'], norm['std']
    
    template_path = config['template_file']
    if not os.path.exists(template_path):
        print("\n" + "="*60)
        print(" [ERROR] Base template model not found!")
        print(" Due to the strict license of the FaceScape dataset, we cannot")
        print(" distribute the original 'template.ply' file.")
        print(" Please apply for the FaceScape dataset from the official website,")
        print(" and place the base topology model here as 'template.ply'.")
        print(" For details, please refer to the README.md document.")
        print("="*60 + "\n")
        sys.exit(1)
        
    template_mesh = trimesh.load(template_path, process=False)
    mask_faces_np = template_mesh.faces 
    edge_index = get_edges_from_faces(mask_faces_np, device)
    sym_indices = get_symmetry_indices(template_mesh.vertices).to(device)

    input_dir = os.path.join(project_root, MASK_input_DIR)
    output_dir = os.path.join(project_root, OUTPUT_DIR_NAME)
    os.makedirs(output_dir, exist_ok=True)
    
    all_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.obj')])
    specific_ids = ["122", "212", "610"]
    target_files = [f for f in all_files if any(sid in f for sid in specific_ids)]
    
    if not target_files:
       print(f"Error: Target sample files (122, 212, 610) not found in {input_dir}")
       return
    
    metrics_table = {mode: {"mse": [], "hd": [], "norm": []} for mode in ABLATION_MODES.keys()} 

    for idx, f_name in enumerate(target_files):
        print(f"Processing sample [{idx+1}/{len(target_files)}]: {f_name}")
        
        target_path = os.path.join(input_dir, f_name)
        try: mesh_obj = trimesh.load(target_path, process=False)
        except: continue
        verts_tensor = torch.from_numpy(mesh_obj.vertices).float().to(device)
        M_gt_norm, _, _ = normalize_to_unit_sphere(verts_tensor)
        
        keep_mask = create_half_face_hole(M_gt_norm) 
        hole_mask = ~keep_mask
        boundary_mask = get_boundary_mask(M_gt_norm.shape[0], keep_mask, edge_index)
        
        temp_gt_mesh = trimesh.Trimesh(vertices=M_gt_norm.cpu().numpy(), faces=mask_faces_np, process=False)
        gt_normals = torch.from_numpy(temp_gt_mesh.vertex_normals).float().to(device)

        input_save_dir = os.path.join(output_dir, "inputs_with_holes")
        os.makedirs(input_save_dir, exist_ok=True)
        save_masked_mesh(M_gt_norm, torch.tensor(mask_faces_np, device=device), keep_mask, os.path.join(input_save_dir, f"input_{f_name}"))

        z_opt = torch.zeros((1, config['z_length']), device=device, requires_grad=True)
        s_opt = torch.tensor([1.0], device=device, requires_grad=True)
        r_opt = torch.zeros(3, device=device, requires_grad=True); t_opt = torch.zeros(3, device=device, requires_grad=True)
        def apply_transform(v, s, r, t):
            cx, cy, cz = torch.cos(r[0]), torch.cos(r[1]), torch.cos(r[2])
            sx, sy, sz = torch.sin(r[0]), torch.sin(r[1]), torch.sin(r[2])
            Rx = torch.tensor([[1,0,0],[0,cx,-sx],[0,sx,cx]], device=v.device)
            Ry = torch.tensor([[cy,0,sy],[0,1,0],[-sy,0,cy]], device=v.device)
            Rz = torch.tensor([[cz,-sz,0],[sz,cz,0],[0,0,1]], device=v.device)
            R = Rz @ Ry @ Rx
            return (v * s) @ R.T + t
        optimizer_s1 = optim.Adam([z_opt, s_opt, r_opt, t_opt], lr=0.01)
        for _ in range(150):
            optimizer_s1.zero_grad()
            M_base = (decoder(z_opt, 1) * std + mean).view(-1, 3) 
            M_base = (M_base - M_base.mean(dim=0)) / M_base.abs().max()
            M_fit = apply_transform(M_base, s_opt, r_opt, t_opt)
            loss = torch.nn.functional.mse_loss(M_fit[keep_mask], M_gt_norm[keep_mask])
            loss.backward(); optimizer_s1.step()
        
        with torch.no_grad():
            M_raw_s1 = (decoder(z_opt, 1) * std + mean).view(-1, 3) 
            M_base = (M_raw_s1 - M_raw_s1.mean(dim=0)) / M_raw_s1.abs().max()
            M_fit_s1 = apply_transform(M_base, s_opt, r_opt, t_opt)
            M_target_full = M_gt_norm[sym_indices]; M_target_full[:, 0] = -M_target_full[:, 0]
        
        M_start = M_fit_s1.detach()

        for mode, mode_settings in ABLATION_MODES.items():
            M_final = run_optimization_stage2(
                M_start, M_gt_norm, M_target_full, edge_index, hole_mask, boundary_mask, 
                mode_settings, SPARSITY_RATIO
            )
            
            temp_mesh = trimesh.Trimesh(vertices=M_final.detach().cpu().numpy(), faces=mask_faces_np, process=False)
            pred_normals = torch.from_numpy(temp_mesh.vertex_normals).float().to(device)
            
            mse = compute_mean_error_mm_local(M_final, M_gt_norm, hole_mask) * 1000 
            hd, norm_err = compute_advanced_metrics(M_final, M_gt_norm, pred_normals, gt_normals, hole_mask)
            
            metrics_table[mode]["mse"].append(mse)
            metrics_table[mode]["hd"].append(hd)
            metrics_table[mode]["norm"].append(norm_err)
            
            print(f"    - Mode [{mode}]: MSE={mse:.4f}, HD={hd:.4f}, Norm={norm_err:.4f}")
            
            mode_dir = os.path.join(output_dir, mode)
            os.makedirs(mode_dir, exist_ok=True)
            trimesh.Trimesh(vertices=M_final.detach().cpu().numpy(), faces=mask_faces_np, process=False).export(
                os.path.join(mode_dir, f"{f_name}")
            )
            
            if mode == "full":
                 gt_dir = os.path.join(output_dir, "ground_truth")
                 os.makedirs(gt_dir, exist_ok=True)
                 trimesh.Trimesh(vertices=M_gt_norm.cpu().numpy(), faces=mask_faces_np, process=False).export(
                    os.path.join(gt_dir, f"{f_name}")
                )

    print("\nExporting final ablation report...")
    csv_path = os.path.join(output_dir, "ablation_metrics_breakdown.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Mode", "Mean Hole Error (mm)", "HD", "Normal"])
        for mode in ABLATION_MODES.keys():
            
            if len(metrics_table[mode]["mse"]) == 0:
                continue # 加上这层保护，防止因为没读到数据导致除以0的报错
                
            avg_mse = sum(metrics_table[mode]["mse"]) / len(metrics_table[mode]["mse"])
            avg_hd = sum(metrics_table[mode]["hd"]) / len(metrics_table[mode]["hd"])
            avg_norm = sum(metrics_table[mode]["norm"]) / len(metrics_table[mode]["norm"])
            writer.writerow([mode, f"{avg_mse:.4f}", f"{avg_hd:.4f}", f"{avg_norm:.4f}"])
            print(f"  {mode:<10} | MSE:{avg_mse:.4f} | HD:{avg_hd:.4f} | Norm:{avg_norm:.4f}")
            
    print("\nAblation study completed. All results and metrics have been saved.")

if __name__ == "__main__":
    run_ablation_study()