import torch
from torch.utils.data import Dataset

class PROTACSet(Dataset):
    def __init__(self, ligase_atom, ligase_bond,
                 target_atom, target_bond,
                 ligase_ligand_atom, ligase_ligand_bond, 
                 target_ligand_atom, target_ligand_bond,
                 smiles, label):
        super().__init__()
        self.ligase_atom = ligase_atom
        self.ligase_bond = ligase_bond
        self.target_atom = target_atom
        self.target_bond = target_bond
        self.ligase_ligand_atom = ligase_ligand_atom
        self.ligase_ligand_bond = ligase_ligand_bond
        self.target_ligand_atom = target_ligand_atom
        self.target_ligand_bond = target_ligand_bond
        self.smiles = smiles
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        ligase_atom = self.ligase_atom[idx]
        ligase_bond = self.ligase_bond[idx]
        target_atom = self.target_atom[idx]
        target_bond = self.target_bond[idx]
        ligase_ligand_atom = self.ligase_ligand_atom[idx]
        ligase_ligand_bond = self.ligase_ligand_bond[idx]
        target_ligand_atom = self.target_ligand_atom[idx]
        target_ligand_bond = self.target_ligand_bond[idx]
        smiles = self.smiles[idx]
        label = self.label[idx]
        sample = {"ligase_atom": torch.tensor(ligase_atom), 
                  "ligase_bond": torch.tensor(ligase_bond, dtype=torch.float), 
                  "target_atom": torch.tensor(target_atom), 
                  "target_bond": torch.tensor(target_bond, dtype=torch.float),
                  "ligase_ligand_atom": torch.tensor(ligase_ligand_atom), 
                  "ligase_ligand_bond": torch.tensor(ligase_ligand_bond, dtype=torch.float), 
                  "target_ligand_atom": torch.tensor(target_ligand_atom), 
                  "target_ligand_bond": torch.tensor(target_ligand_bond, dtype=torch.float),
                  "smiles": torch.tensor(smiles), 
                  "label": torch.tensor(label, dtype=torch.long)}
        return sample
