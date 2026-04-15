import os
import torch
import numpy as np
import trimesh
import configparser
import sys
from scipy.spatial import cKDTree

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
sys.path.append(project_root)

from funcs import load_generator

RESULTS_ROOT = "results_comparison" 
CORRECT_CHECKPOINT_DIR = "checkpoints"

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

def get_hole_mask(mesh_gt, mesh_input):
    tree = cKDTree(mesh_input.vertices)
    dists, _ = tree.query(mesh_gt.vertices, k=1)
    hole_mask = dists > 1e-4
    return hole_mask

def simple_laplacian_smooth(mesh, mask, iterations=20):
    neighbors = mesh.vertex_neighbors
    verts = mesh.vertices.copy()
    for _ in range(iterations):
        new_verts = verts.copy()
        indices_to_smooth = np.where(mask)[0]
        for idx in indices_to_smooth:
            nbrs = neighbors[idx]
            if len(nbrs) > 0:
                new_verts[idx] = verts[nbrs].mean(axis=0)
        verts = new_verts
    return verts

def run_extra_baselines():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Info: Starting extra baselines generation (Device: {device})")

    cfg_path = os.path.join(project_root, 'config/config.cfg')
    config = LocalConfig(cfg_path)
    decoder = load_generator(config).to(device).eval()
    
    norm = torch.load(os.path.join(project_root, 'checkpoints/norm.pt'), map_location=device)
    mean_face_tensor = norm['mean'].view(-1, 3).to(device)

    import glob
    sub_dirs = sorted(glob.glob(os.path.join(RESULTS_ROOT, "result_*")))
    
    for d in sub_dirs:
        folder_name = os.path.basename(d)
        p_gt = os.path.join(d, "0_ground_truth.obj")
        p_input = os.path.join(d, "1_input_hole.obj")
        p_stage1 = os.path.join(d, "baseline_stage1.obj")

        if not (os.path.exists(p_gt) and os.path.exists(p_input)): 
            continue

        print(f"Processing: {folder_name}")
        mesh_gt = trimesh.load(p_gt, process=False, force='mesh')
        mesh_input = trimesh.load(p_input, process=False, force='mesh')
        
        if os.path.exists(p_stage1):
            mesh_stage1 = trimesh.load(p_stage1, process=False, force='mesh')
            init_fill_verts = mesh_stage1.vertices
        else:
            init_fill_verts = mesh_gt.vertices
            
        hole_mask = get_hole_mask(mesh_gt, mesh_input)
        mean_mesh_raw = trimesh.Trimesh(vertices=mean_face_tensor.cpu().numpy(), faces=mesh_gt.faces, process=False)
        valid_mask = ~hole_mask
        src_pts = mean_mesh_raw.vertices[valid_mask]
        dst_pts = mesh_gt.vertices[valid_mask]
        
        try:
            T, _, _ = trimesh.registration.procrustes(src_pts, dst_pts)
            mean_mesh_raw.apply_transform(T)
            verts_mean_patch = mesh_gt.vertices.copy()
            verts_mean_patch[hole_mask] = mean_mesh_raw.vertices[hole_mask]
            trimesh.Trimesh(vertices=verts_mean_patch, faces=mesh_gt.faces, process=False).export(
                os.path.join(d, "baseline_mean.obj"))
        except:
            print(f"Warning: Procrustes alignment failed for {folder_name}. Skipping mean baseline.")

        verts_geo = mesh_gt.vertices.copy()
        verts_geo[hole_mask] = init_fill_verts[hole_mask]
        temp_mesh = trimesh.Trimesh(vertices=verts_geo, faces=mesh_gt.faces, process=False)
        smoothed_verts = simple_laplacian_smooth(temp_mesh, hole_mask, iterations=20)
        trimesh.Trimesh(vertices=smoothed_verts, faces=mesh_gt.faces, process=False).export(
            os.path.join(d, "baseline_geometric.obj"))

    print("\nSuccess: Extra baselines (baseline_mean.obj, baseline_geometric.obj) have been generated.")

if __name__ == "__main__":
    run_extra_baselines()