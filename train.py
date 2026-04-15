import os
import sys
import argparse
import warnings
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
sys.path.append(project_root)

from config.config import read_config
from models import FMGenModel
from funcs import get_mesh_matrices, spherical_regularization_loss
from my_dataset import FaceDataset 

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def load_model(config):
    pA, pD, pU = get_mesh_matrices(config)
    model = FMGenModel(config, pA, pD, pU)
    return model

def train_epoch(model, train_loader, optimizer, device, lambda_reg=1.0):
    model.train()
    total_loss_l1 = 0
    total_loss_mse = 0
    total_loss_reg = 0
    num_graphs = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()

        out, z = model(batch)

        loss_mse = F.mse_loss(out, batch.y)
        loss_l1 = F.l1_loss(out, batch.y)
        loss_reg = spherical_regularization_loss(z)

        loss = loss_l1 + loss_mse + lambda_reg * loss_reg

        loss.backward()
        optimizer.step()

        bs = batch.num_graphs
        total_loss_mse += bs * loss_mse.item()
        total_loss_l1 += bs * loss_l1.item()
        total_loss_reg += bs * loss_reg.item()
        num_graphs += bs

    return total_loss_l1 / num_graphs, total_loss_mse / num_graphs, total_loss_reg / num_graphs

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Info: Using device: {device}")

    if not os.path.exists(config['checkpoint_dir']):
        os.makedirs(config['checkpoint_dir'])

    print("Info: Loading dataset...")
    data_root = config.get('dataset_dir', 'dataset') 
    dataset_train = FaceDataset(root=data_root, split='train')
    data_len = len(dataset_train)
    
    real_batch_size = min(config['batch_size'], 4)
    if data_len < real_batch_size:
        real_batch_size = 2
    
    print(f"Info: Dataset size: {data_len}, Batch size: {real_batch_size}")

    train_loader = DataLoader(dataset_train, batch_size=real_batch_size, shuffle=True, num_workers=0)
    dataset_test = FaceDataset(root=data_root, split='test')
    test_loader = DataLoader(dataset_test, batch_size=real_batch_size, shuffle=False, num_workers=0)

    print("Info: Initializing model...")
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

    model = load_model(config)
    model.to(device)

    print("Info: Checking for checkpoints...")
    enc_path = os.path.join(config['checkpoint_dir'], 'checkpoint_encoder.pt')
    dec_path = os.path.join(config['checkpoint_dir'], 'checkpoint_decoder.pt')

    if os.path.exists(enc_path):
        try:
            model.encoder.load_state_dict(torch.load(enc_path))
            print(f"Info: Encoder loaded from {enc_path}")
        except Exception as e:
            print(f"Warning: Failed to load encoder: {e}")

    if os.path.exists(dec_path):
        try:
            model.decoder.load_state_dict(torch.load(dec_path))
            print(f"Info: Decoder loaded from {dec_path}. Resuming training...")
        except Exception as e:
            print(f"Warning: Failed to load decoder: {e}")
    else:
        print("Info: No checkpoint found. Starting from scratch.")

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    print("Info: Starting training loop...")
    best_loss = float('inf')
    lambda_reg = config['lambda_reg']

    for epoch in range(1, config['epoch'] + 1):
        current_lr = scheduler.get_last_lr()[0]
        l1, mse, reg = train_epoch(model, train_loader, optimizer, device, lambda_reg)
        
        print(f"Epoch {epoch:03d} | LR: {current_lr:.6f} | L1: {l1:.5f} | MSE: {mse:.5f} | Reg: {reg:.5f}")

        scheduler.step()

        current_total_loss = l1 + mse + reg
        if current_total_loss < best_loss:
            best_loss = current_total_loss
            torch.save(model.encoder.state_dict(), os.path.join(config['checkpoint_dir'], 'checkpoint_encoder.pt'))
            torch.save(model.decoder.state_dict(), os.path.join(config['checkpoint_dir'], 'checkpoint_decoder.pt'))

    print("\nTraining completed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='config/config.cfg', help="Path to config file")
    args = parser.parse_args()

    print(f"Info: Reading config from {args.config_file}")
    config = read_config(args.config_file)
    
    
    train(config)

if __name__ == "__main__":
    main()