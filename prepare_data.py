import torch
import pickle
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from rdkit import Chem
from pathlib import  Path


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
                "label.pt",
                ]

    def process(self):
        with open('name.pkl','rb') as f:
            name_list = pickle.load(f)
        
        ligase_ligand = []
        for name in name_list:
            graph = mol2graph('ligase_ligand/'+name+".mol2", LIGAND_ATOM_TYPE)
            ligase_ligand.append(graph)
        data, slices = self.collate(ligase_ligand)
        torch.save((data, slices), self.processed_paths[0])

        ligase_pocket = []
        for name in name_list:
            graph = mol2graph('ligase_pocket_5/'+name+".mol2", PROTEIN_ATOM_TYPE)
            ligase_pocket.append(graph)
        data, slices = self.collate(ligase_pocket)
        torch.save((data, slices), self.processed_paths[1])

        target_ligand = []
        for name in name_list:
            graph = mol2graph('target_ligand/'+name+".mol2", LIGAND_ATOM_TYPE)
            target_ligand.append(graph)
        data, slices = self.collate(target_ligand)
        torch.save((data, slices), self.processed_paths[2])

        target_pocket = []
        for name in name_list:
            graph = mol2graph('target_pocket_5/'+name+".mol2", PROTEIN_ATOM_TYPE)
            target_pocket.append(graph)
        data, slices = self.collate(target_pocket)
        torch.save((data, slices), self.processed_paths[3])

        smiles = []
        for i in name_list:
            smi_num = i.split("_")[0]
            if Path("protacs/"+i+"/linker_"+smi_num+".smi").exists():
                with open("protacs/"+i+"/linker_"+smi_num+".smi") as f:
                    smi = f.read()
                smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
                smiles.append(trans_smiles(smi.strip()))
            else:
                smiles.append([0])
        with open(self.processed_paths[4],"wb") as f:
            pickle.dump(smiles,f)

        label_csv = pd.read_csv("protacs.csv")
        id = list(label_csv["Compound ID"])
        tar = list(label_csv["Target"])
        e3  = list(label_csv["E3 Ligase"])
        lab1 = list(label_csv["Degradation Identification new 1"])
        lab2 = list(label_csv["Degradation Identification new 2"])
        labels1 = {}
        labels2 = {}

        for i in range(len(id)):
            a = str(id[i])+"_"+tar[i].split('_')[0].replace(' ','-').replace('/','-')+"_"+e3[i].split('_')[0]
            labels1[a] = lab1[i]
            labels2[a] = lab2[i]

        label1 = []
        for i in name_list:
            if labels1[i]=='Good':
                label1.append(1)
            elif  labels1[i]=='Bad':
                label1.append(0)
        torch.save(label1, self.processed_paths[5])



if __name__=="__main__":
    ligase_ligand = GraphData("ligase_ligand")
