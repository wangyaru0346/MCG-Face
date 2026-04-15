import torch
import torch.optim as optim
import os
import sys
import numpy as np
import configparser
import trimesh
from scipy.spatial import cKDTree

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
sys.path.append(project_root)

from funcs import load_generator

CORRECT_CHECKPOINT_DIR = "checkpoints"
MASK_input_DIR = "examples" 
OUTPUT_DIR_NAME = "results_comparison"

def compute_gradient_loss_3d(current_verts, target_verts, edge_index, hole_mask):
    src, dst = edge_index[0], edge_index[1]
    curr_edges = current_verts[src] - current_verts[dst]
    target_edges = target_verts[src] - target_verts[dst]
    mask_edges = hole_mask[src] | hole_mask[dst]
    if mask_edges.sum() == 0: return torch.tensor(0.0, device=current_verts.device)
    return torch.mean((curr_edges[mask_edges] - target_edges[mask_edges]) ** 2)

def project_points(verts):
    return verts[:, :2]

def compute_prdl_gradient_2d(current_verts, target_verts, edge_index, hole_mask):
    curr_2d = project_points(current_verts)
    target_2d = project_points(target_verts)
    src, dst = edge_index[0], edge_index[1]
    curr_edges = curr_2d[src] - curr_2d[dst]
    target_edges = target_2d[src] - target_2d[dst]
    mask_edges = hole_mask[src] | hole_mask[dst]
    if mask_edges.sum() == 0: return torch.tensor(0.0, device=current_verts.device)
    return torch.mean((curr_edges[mask_edges] - target_edges[mask_edges]) ** 2)

def get_symmetry_indices(verts_np):
    flipped_verts = verts_np.copy()
    flipped_verts[:, 0] = -flipped_verts[:, 0]
    tree = cKDTree(verts_np)
    dists, indices = tree.query(flipped_verts, k=1)
    return torch.from_numpy(indices).long()

def get_edges_from_faces(faces_np, device):
    edges = set()
    for f in faces_np:
        edges.add(tuple(sorted((f[0], f[1]))))
        edges.add(tuple(sorted((f[1], f[2]))))
        edges.add(tuple(sorted((f[2], f[0]))))
    directed = []
    for u, v in edges:
        directed.append([u, v])
        directed.append([v, u])
    return torch.tensor(directed, dtype=torch.long).t().contiguous().to(device)

def get_boundary_mask(n_verts, keep_mask, edge_index):
    boundary = torch.zeros(n_verts, dtype=torch.bool, device=keep_mask.device)
    src, dst = edge_index[0], edge_index[1]
    is_boundary = keep_mask[src] & (~keep_mask[dst])
    boundary[src[is_boundary]] = True
    return boundary

def normalize_to_unit_sphere(verts):
    if isinstance(verts, np.ndarray): verts = torch.from_numpy(verts).float()
    center = verts.mean(dim=0, keepdim=True)
    verts_c = verts - center
    scale = verts_c.abs().max()
    if scale < 1e-6: scale = torch.tensor(1.0)
    return verts_c / scale, center, scale

def save_masked_mesh(verts, faces, keep_mask, path):
    if torch.is_tensor(verts): verts = verts.detach().cpu().numpy()
    if torch.is_tensor(faces): faces = faces.detach().cpu().numpy()
    if torch.is_tensor(keep_mask): keep_mask = keep_mask.detach().cpu().numpy()
    valid_faces_idx = keep_mask[faces[:, 0]] & keep_mask[faces[:, 1]] & keep_mask[faces[:, 2]]
    new_faces = faces[valid_faces_idx]
    old_to_new_map = -np.ones(len(verts), dtype=int)
    valid_verts_idx = np.where(keep_mask)[0]
    old_to_new_map[valid_verts_idx] = np.arange(len(valid_verts_idx))
    new_faces_remapped = old_to_new_map[new_faces]
    new_verts = verts[keep_mask]
    trimesh.Trimesh(vertices=new_verts, faces=new_faces_remapped, process=False).export(path)

def create_varied_holes(verts_tensor, mode='random'):
    n_verts = verts_tensor.shape[0]
    device = verts_tensor.device
    keep_mask = torch.ones(n_verts, dtype=torch.bool, device=device)
    x, y, z = verts_tensor[:, 0], verts_tensor[:, 1], verts_tensor[:, 2]
    avg_z = z.mean()

    def dig_sphere(center_v, rad):
        dists = torch.norm(verts_tensor - center_v, dim=1)
        edge_noise = (torch.sin(x*30) + torch.cos(y*30)) * 0.02 * rad
        return (dists + edge_noise) > rad

    def get_noise_mask(freq=10.0, threshold=0.2):
        noise = torch.sin(x * freq) + torch.cos(y * freq * 1.5) + torch.sin(z * freq * 2.0)
        noise = noise / noise.abs().max() 
        noise += torch.randn_like(x) * 0.3
        return noise > threshold

    available_modes = [
        'eyes_mouth', 'half_left', 
        'chin_cut', 'forehead_cut', 'diagonal_cut', 
        'random_patches_small', 'random_patches_large', 'noise_erosion', 
        'mixed_complex'
    ]

    if mode == 'random':
        weights = torch.ones(len(available_modes))
        weights[available_modes.index('random_patches_small')] = 2.0
        weights[available_modes.index('noise_erosion')] = 2.0
        idx = torch.multinomial(weights, 1).item()
        mode = available_modes[idx]

    print(f"    Info: Applied hole mode [{mode.upper()}]")
    
    if mode == 'eyes_mouth':
        keep_mask &= dig_sphere(torch.tensor([-0.28, 0.22, avg_z+0.05], device=device), 0.16)
        keep_mask &= dig_sphere(torch.tensor([0.28, 0.22, avg_z+0.05], device=device), 0.16)
        keep_mask &= dig_sphere(torch.tensor([0.0, -0.38, avg_z+0.05], device=device), 0.22)
    elif mode == 'half_left':
        keep_mask &= x > 0.02
    elif mode == 'chin_cut':
        keep_mask &= y > -0.35
    elif mode == 'forehead_cut':
        keep_mask &= y < 0.35
    elif mode == 'diagonal_cut':
        keep_mask &= (x + y*0.5) > -0.1
    elif 'random_patches' in mode:
        num = torch.randint(6, 15, (1,)).item() if 'small' in mode else torch.randint(3, 6, (1,)).item()
        base_rad = 0.08 if 'small' in mode else 0.15
        for _ in range(num):
            center_idx = torch.randint(0, n_verts, (1,)).item()
            rad = base_rad * (torch.rand(1).item() + 0.5)
            keep_mask &= dig_sphere(verts_tensor[center_idx], rad)
    elif mode == 'noise_erosion':
        keep_mask &= get_noise_mask(freq=8.0, threshold=0.1)
    elif mode == 'mixed_complex':
        keep_mask &= dig_sphere(torch.tensor([0.0, -0.38, avg_z+0.05], device=device), 0.22)
        for _ in range(4):
            idx = torch.randint(0, n_verts, (1,)).item()
            if abs(verts_tensor[idx, 0]) > 0.2:
                 keep_mask &= dig_sphere(verts_tensor[idx], 0.1)

    if keep_mask.sum() / n_verts < 0.30:
        center_idx = torch.randint(0, n_verts, (1,)).item()
        keep_mask = dig_sphere(verts_tensor[center_idx], 0.3)
        mode = 'fallback'

    return keep_mask, mode

class LocalConfig:
    def __init__(self, config_path):
        self.conf = {}
        cp = configparser.ConfigParser()
        cp.read(config_path)
        self.conf['z_length'] = cp.getint('model parameters', 'z_length')
        self.conf['batch_norm'] = cp.getboolean('model parameters', 'batch_norm')
        self.conf['num_features_global'] = [int(x) for x in cp.get('model parameters', 'num_features_global').split(',')]
        self.conf['num_features_local'] = [int(x) for x in cp.get('model parameters', 'num_features_local').split(',')]
        self.conf['down_sampling_factors'] = [float(x) for x in cp.get('model parameters', 'down_sampling_factors').split(',')]
        self.conf['n_layers'] = len(self.conf['num_features_global']) - 1
        self.conf['template_file'] = os.path.join(project_root, cp.get('I/O parameters', 'template_file'))
        self.conf['checkpoint_dir'] = CORRECT_CHECKPOINT_DIR
    def __getitem__(self, key): return self.conf[key]

def run_batch_final_submission():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Info: Starting comparative experiment (Device: {device})")

    cfg_path = os.path.join(project_root, 'config/config.cfg')
    config = LocalConfig(cfg_path)
    ckpt_path = os.path.join(config['checkpoint_dir'], 'checkpoint_decoder.pt')
    if not os.path.exists(ckpt_path): 
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

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
    
    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.obj')])[:10]
    torch.manual_seed(42)

    for idx, f_name in enumerate(files):
        print(f"Processing sample [{idx+1}/{len(files)}]: {f_name}")
        sample_dir = os.path.join(output_dir, f"result_{f_name[:-4]}")
        os.makedirs(sample_dir, exist_ok=True)
        
        target_path = os.path.join(input_dir, f_name)
        try: mesh_obj = trimesh.load(target_path, process=False)
        except: continue
        
        verts_tensor = torch.from_numpy(mesh_obj.vertices).float().to(device)
        M_gt_norm, _, _ = normalize_to_unit_sphere(verts_tensor)
        
        keep_mask, mode = create_varied_holes(M_gt_norm, mode='random')
        hole_mask = ~keep_mask
        boundary_mask = get_boundary_mask(M_gt_norm.shape[0], keep_mask, edge_index)

        with torch.no_grad():
            M_target_full = M_gt_norm[sym_indices] 
            M_target_full[:, 0] = -M_target_full[:, 0]
            M_baseline_sym = M_gt_norm.clone()
            M_baseline_sym[hole_mask] = M_target_full[hole_mask]
            
            trimesh.Trimesh(
                vertices=M_baseline_sym.cpu().numpy(), 
                faces=mask_faces_np, 
                process=False
            ).export(os.path.join(sample_dir, "baseline_symmetry.obj"))

        z_opt = torch.zeros((1, config['z_length']), device=device, requires_grad=True)
        s_opt = torch.tensor([1.0], device=device, requires_grad=True)
        r_opt = torch.zeros(3, device=device, requires_grad=True)
        t_opt = torch.zeros(3, device=device, requires_grad=True)
        def apply_transform(v, s, r, t):
            cx, cy, cz = torch.cos(r[0]), torch.cos(r[1]), torch.cos(r[2])
            sx, sy, sz = torch.sin(r[0]), torch.sin(r[1]), torch.sin(r[2])
            Rx = torch.tensor([[1,0,0],[0,cx,-sx],[0,sx,cx]], device=v.device)
            Ry = torch.tensor([[cy,0,sy],[0,1,0],[-sy,0,cy]], device=v.device)
            Rz = torch.tensor([[cz,-sz,0],[sz,cz,0],[0,0,1]], device=v.device)
            R = Rz @ Ry @ Rx
            return (v * s) @ R.T + t
        optimizer_s1 = optim.Adam([z_opt, s_opt, r_opt, t_opt], lr=0.01)
        for _ in range(250):
            optimizer_s1.zero_grad()
            M_base = (decoder(z_opt, 1) * std + mean).view(-1, 3) 
            M_base = (M_base - M_base.mean(dim=0)) / M_base.abs().max()
            M_fit = apply_transform(M_base, s_opt, r_opt, t_opt)
            loss_fit = torch.nn.functional.mse_loss(M_fit[keep_mask], M_gt_norm[keep_mask])
            loss = 100.0 * loss_fit + 1.0 * (torch.norm(z_opt)-1.0)**2
            loss.backward()
            optimizer_s1.step()

        with torch.no_grad():
            M_raw_s1 = (decoder(z_opt, 1) * std + mean).view(-1, 3) 
            M_base = (M_raw_s1 - M_raw_s1.mean(dim=0)) / M_raw_s1.abs().max()
            M_fit_s1 = apply_transform(M_base, s_opt, r_opt, t_opt)
            trimesh.Trimesh(
                vertices=M_fit_s1.cpu().numpy(), 
                faces=mask_faces_np,
                process=False
            ).export(os.path.join(sample_dir, "baseline_stage1.obj"))

            M_target_full = M_gt_norm[sym_indices] 
            M_target_full[:, 0] = -M_target_full[:, 0] 

        delta = torch.zeros_like(M_fit_s1, requires_grad=True, device=device)
        optimizer_s2 = optim.Adam([delta], lr=0.005) 
        weight_map_fit = torch.ones_like(keep_mask, dtype=torch.float32)
        weight_map_fit[keep_mask] = 5000.0 
        weight_map_fit[boundary_mask] = 5000.0 

        for step in range(400):
            optimizer_s2.zero_grad()
            M_curr = M_fit_s1 + delta
            loss_fit = (torch.sum((M_curr - M_gt_norm)**2, dim=1) * weight_map_fit).mean()
            loss_grad_3d = compute_gradient_loss_3d(M_curr, M_target_full, edge_index, hole_mask)
            loss_grad_2d = compute_prdl_gradient_2d(M_curr, M_target_full, edge_index, hole_mask)
            l_smooth = compute_gradient_loss_3d(M_curr, M_curr, edge_index, hole_mask)
            l_decay = torch.mean(delta**2)
            loss = 1.0 * loss_fit + 500.0 * loss_grad_3d + 300.0 * loss_grad_2d + 10.0 * l_smooth + 1.0 * l_decay
            loss.backward()
            optimizer_s2.step()

        with torch.no_grad():
            M_final = M_fit_s1 + delta
            trimesh.Trimesh(
                vertices=M_final.cpu().numpy(), 
                faces=mask_faces_np,
                process=False
            ).export(os.path.join(sample_dir, "2_reconstructed.obj"))
            
            trimesh.Trimesh(
                vertices=M_gt_norm.cpu().numpy(), 
                faces=mask_faces_np,
                process=False
            ).export(os.path.join(sample_dir, "0_ground_truth.obj"))
            
            save_masked_mesh(M_gt_norm, torch.tensor(mask_faces_np, device=device), keep_mask, os.path.join(sample_dir, "1_input_hole.obj"))

    print(f"\nComparative experiment process completed. Results saved to: {OUTPUT_DIR_NAME}")

if __name__ == "__main__":
    run_batch_final_submission()