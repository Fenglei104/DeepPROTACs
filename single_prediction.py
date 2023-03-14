import torch
import os
from rdkit import Chem
from torch_geometric.data import Batch
import sys


path = sys.argv[1]

os.system(f"babel {path}/ligase_ligand.mol2 {path}/ligase_ligand.mol2 >/dev/null 2>&1")
os.system(f"babel {path}/ligase_pocket.mol2 {path}/ligase_pocket.mol2 >/dev/null 2>&1")
os.system(f"babel {path}/target_ligand.mol2 {path}/target_ligand.mol2 >/dev/null 2>&1")
os.system(f"babel {path}/target_pocket.mol2 {path}/target_pocket.mol2 >/dev/null 2>&1")

import torch
import pickle
from torch.utils.data import DataLoader
from torch_geometric.data import InMemoryDataset, Data
from rdkit import Chem

PROTEIN_ATOM_TYPE =['C','N','O','S']
LIGAND_ATOM_TYPE = ['C','N','O','S','F','Cl','Br','I','P']
SMILES_CHAR =['[PAD]', 'C', '(', '=', 'O', ')', 'N', '[', '@', 'H', ']', '1', 'c', 'n', '/', '2', '#', 'S', 's', '+', '-', '\\', '3', '4', 'l', 'F', 'o', 'I', 'B', 'r', 'P', '5', '6', 'i', '7', '8', '9', '%', '0', 'p']
EDGE_ATTR = {'1':1,'2':2,'3':3,'ar':4,'am':5}
def trans_smiles(x):
    temp = list(x)
    temp = [SMILES_CHAR.index(i) if i in SMILES_CHAR else len(SMILES_CHAR) for i in temp]
    return temp

def mol2graph(path, ATOM_TYPE):
    with open(path) as f:
        lines = f.readlines()
    atom_lines = lines[lines.index('@<TRIPOS>ATOM\n')+1:lines.index('@<TRIPOS>BOND\n')]
    bond_lines = lines[lines.index('@<TRIPOS>BOND\n')+1:]
    atoms = []
    for atom in atom_lines:
        ele = atom.split()[5].split('.')[0]
        atoms.append(ATOM_TYPE.index(ele) 
                        if ele in ATOM_TYPE 
                        else len(ATOM_TYPE))
    edge_1 = [int(i.split()[1])-1 for i in bond_lines]
    edge_2 = [int(i.split()[2])-1 for i in bond_lines]
    edge_attr = [EDGE_ATTR[i.split()[3]] for i in bond_lines]
    x = torch.tensor(atoms)
    edge_idx=torch.tensor([edge_1+edge_2,edge_2+edge_1])
    edge_attr=torch.tensor(edge_attr+edge_attr)
    graph = Data(x=x, edge_index=edge_idx, edge_attr=edge_attr)
    return graph

class GraphData(InMemoryDataset):
    def __init__(self, name, root="data"):
        super().__init__(root)
        if name == "ligase_ligand":
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif name == "ligase_pocket":
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif name == "target_ligand":
            self.data, self.slices = torch.load(self.processed_paths[2])
        elif name == "target_pocket":
            self.data, self.slices = torch.load(self.processed_paths[3])

    @property
    def processed_file_names(self):
        return ["ligase_ligand.pt",
                "ligase_pocket.pt",
                "target_ligand.pt",
                "target_pocket.pt",
                "smiles.pkl",
                ]

    def process(self):

        graph = mol2graph(f"{path}/ligase_ligand.mol2", LIGAND_ATOM_TYPE)
        ligase_ligand = [graph]
        data, slices = self.collate(ligase_ligand)
        torch.save((data, slices), self.processed_paths[0])

        graph = mol2graph(f"{path}/ligase_pocket.mol2", PROTEIN_ATOM_TYPE)
        ligase_pocket = [graph]
        data, slices = self.collate(ligase_pocket)
        torch.save((data, slices), self.processed_paths[1])

        graph = mol2graph(f"{path}/target_ligand.mol2", LIGAND_ATOM_TYPE)
        target_ligand = [graph]
        data, slices = self.collate(target_ligand)
        torch.save((data, slices), self.processed_paths[2])

        graph = mol2graph(f"{path}/target_pocket.mol2", PROTEIN_ATOM_TYPE)
        target_pocket = [graph]
        data, slices = self.collate(target_pocket)
        torch.save((data, slices), self.processed_paths[3])

        with open(f"{path}/linker.smi") as f:
            smi = f.read()
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        smiles = [trans_smiles(smi.strip())]
        with open(self.processed_paths[4],"wb") as f:
            pickle.dump(smiles,f)
    
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


def collater(data_list):
    batch = {}
    ligase_ligand = [x["ligase_ligand"] for x in data_list]
    ligase_pocket = [x["ligase_pocket"] for x in data_list]
    target_ligand = [x["target_ligand"] for x in data_list]
    target_pocket = [x["target_pocket"] for x in data_list]
    smiles = [torch.tensor(x["smiles"]) for x in data_list]
    smiles_length = [len(x["smiles"]) for x in data_list]

    batch["ligase_ligand"] = Batch.from_data_list(ligase_ligand)
    batch["ligase_pocket"] = Batch.from_data_list(ligase_pocket)
    batch["target_ligand"] = Batch.from_data_list(target_ligand)
    batch["target_pocket"] = Batch.from_data_list(target_pocket)
    batch["smiles"] = torch.nn.utils.rnn.pad_sequence(smiles, batch_first=True)
    batch["smiles_length"] = smiles_length
    return batch


class PROTACSet(Dataset):
    def __init__(self, ligase_ligand, ligase_pocket, target_ligand, target_pocket, smiles):
        super().__init__()
        self.ligase_ligand = ligase_ligand
        self.ligase_pocket = ligase_pocket
        self.target_ligand = target_ligand
        self.target_pocket = target_pocket
        self.smiles = smiles


    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        sample = {
            "ligase_ligand": self.ligase_ligand[idx],
            "ligase_pocket": self.ligase_pocket[idx],
            "target_ligand": self.target_ligand[idx],
            "target_pocket": self.target_pocket[idx],
            "smiles": self.smiles[idx],
        }
        return sample

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = f"{path}/data"
ligase_ligand = GraphData("ligase_ligand",root)
ligase_pocket = GraphData("ligase_pocket",root)
target_ligand = GraphData("target_ligand",root)
target_pocket = GraphData("target_pocket",root)
with open(os.path.join(target_pocket.processed_dir, "smiles.pkl"),"rb") as f:
    smiles = pickle.load(f)

test_set = PROTACSet(
    ligase_ligand, 
    ligase_pocket,
    target_ligand, 
    target_pocket, 
    smiles,
)

testloader = DataLoader(test_set, batch_size=1, collate_fn=collater)

model = torch.load('model/test.pt')
for data_sample in testloader:
    outputs = model(data_sample['ligase_ligand'].to(device),
                            data_sample['ligase_pocket'].to(device),
                            data_sample['target_ligand'].to(device),
                            data_sample['target_pocket'].to(device),
                            data_sample['smiles'].to(device),
                            data_sample['smiles_length'],)

    pred_y = torch.max(outputs,1)[1].item()
    print(pred_y)
