import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GraphConv(nn.Module):
    def __init__(self, num_embeddings):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim = 64)
        self.gcn1 = GCNConv(64, 128)
        self.gcn2 = GCNConv(128, 64)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr.to(torch.float)
        x = self.embed(x)
        x = self.gcn1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.gcn2(x, edge_index, edge_attr)
        x = global_max_pool(x, batch)
        return x

class SmilesNet(nn.Module):
    def __init__(self, batch_size = 1):
        super().__init__()
        self.batch_size = batch_size
        self.embed = nn.Embedding(41, 64, padding_idx=0)
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 64)

    def forward(self, x, s):
        x = self.embed(x)
        x = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        out, (h, c) = self.lstm(x, None)
        out, _ = pad_packed_sequence(out, batch_first=True)
        y = self.fc(out[:,-1,:])
        return y

class ProtacModel(nn.Module):
    def __init__(self, 
                 ligase_ligand_model, 
                 ligase_pocket_model,
                 target_ligand_model, 
                 target_pocket_model, 
                 smiles_model):
        
        super().__init__()
        self.ligase_ligand_model = ligase_ligand_model
        self.ligase_pocket_model = ligase_pocket_model
        self.target_ligand_model = target_ligand_model
        self.target_pocket_model = target_pocket_model
        self.smiles_model = smiles_model
        self.fc1 = nn.Linear(64*5,64)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(64,2)

    def forward(self,
                ligase_ligand,
                ligase_pocket,
                target_ligand,
                target_pocket,
                smiles,
                smiles_length,):
        v_0 = self.ligase_ligand_model(ligase_ligand)
        v_1 = self.ligase_pocket_model(ligase_pocket)
        v_2 = self.target_ligand_model(target_ligand)
        v_3 = self.target_pocket_model(target_pocket)
        v_4 = self.smiles_model(smiles, smiles_length)
        v_f = torch.cat((v_0, v_1, v_2, v_3, v_4), 1)
        v_f = self.relu(self.fc1(v_f))
        v_f = self.fc2(v_f)
        return v_f

