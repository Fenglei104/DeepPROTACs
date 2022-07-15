import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


def collater(data_list):
    batch = {}
    name = [x["name"] for x in data_list]
    ligase_ligand = [x["ligase_ligand"] for x in data_list]
    ligase_pocket = [x["ligase_pocket"] for x in data_list]
    target_ligand = [x["target_ligand"] for x in data_list]
    target_pocket = [x["target_pocket"] for x in data_list]
    smiles = [torch.tensor(x["smiles"]) for x in data_list]
    smiles_length = [len(x["smiles"]) for x in data_list]
    label = [x["label"] for x in data_list]

    batch["name"] = name
    batch["ligase_ligand"] = Batch.from_data_list(ligase_ligand)
    batch["ligase_pocket"] = Batch.from_data_list(ligase_pocket)
    batch["target_ligand"] = Batch.from_data_list(target_ligand)
    batch["target_pocket"] = Batch.from_data_list(target_pocket)
    batch["smiles"] = torch.nn.utils.rnn.pad_sequence(smiles, batch_first=True)
    batch["smiles_length"] = smiles_length
    batch["label"]=torch.tensor(label)
    return batch


class PROTACSet(Dataset):
    def __init__(self, name, ligase_ligand, ligase_pocket, target_ligand, target_pocket, smiles, label):
        super().__init__()
        self.name = name
        self.ligase_ligand = ligase_ligand
        self.ligase_pocket = ligase_pocket
        self.target_ligand = target_ligand
        self.target_pocket = target_pocket
        self.smiles = smiles
        self.label = label


    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        sample = {
            "name": self.name[idx],
            "ligase_ligand": self.ligase_ligand[idx],
            "ligase_pocket": self.ligase_pocket[idx],
            "target_ligand": self.target_ligand[idx],
            "target_pocket": self.target_pocket[idx],
            "smiles": self.smiles[idx],
            "label": self.label[idx],
        }
        return sample


