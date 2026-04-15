import torch
from torch_geometric.data import Data, Dataset
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir

class FaceDataset(Dataset):
    def __init__(self, root, split='train'):
        """
        root: Points to the data directory (e.g., 'dataset')
        split: 'train' or 'test'
        """
        super().__init__(root)
        self.split = split
        
        if split == 'train':
            self.data_path = os.path.join(root, 'facescape_train_data.pt')
        else:
            test_path = os.path.join(root, 'facescape_test_data.pt')
            if os.path.exists(test_path):
                self.data_path = test_path
            else:
                self.data_path = os.path.join(root, 'facescape_train_data.pt')

        if not os.path.exists(self.data_path):
            print("\n" + "="*60)
            print(f" [ERROR] Training data not found at: {self.data_path}")
            print(" Due to the strict license of the FaceScape dataset, we cannot")
            print(" distribute the processed training data files (.pt).")
            print(" Please apply for the FaceScape dataset, process it into '.pt' format,")
            print(" and place it in the 'dataset/' directory.")
            print(" For details, please refer to the README.md document.")
            print("="*60 + "\n")
            sys.exit(1)
            
        self.data = torch.load(self.data_path)
        
        norm_path = os.path.join(project_root, 'checkpoints', 'norm.pt')
        
        if os.path.exists(norm_path):
            norm_dict = torch.load(norm_path)
            self.mean = norm_dict['mean']
            self.std = norm_dict['std']
            print(f"Info: [{split}] Norm data loaded successfully.")
        else:
            self.mean = torch.zeros(1, 12596, 3)
            self.std = torch.ones(1, 12596, 3)
            print(f"Warning: [{split}] norm.pt not found. Data will NOT be normalized!")

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        pos = self.data[idx]
        
        x = (pos - self.mean) / (self.std + 1e-8)
        
        data = Data(x=x, y=x, pos=pos)
        
        return data