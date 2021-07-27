import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.projection = nn.Linear(c_in, c_out)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
    
    def forward(self, node_feats, adj_matrix):
        """ node_feats : [batch_size, num_nodes, c_in]
            adj_matrix : [batch_size, num_nodes, num_nodes]
        """
        num_neighbours = adj_matrix.sum(dim=-1)
        lap = torch.diag(torch.pow(num_neighbours,-0.5)) * adj_matrix * torch.diag(torch.pow(num_neighbours,-0.5))
        node_feats = self.projection(node_feats)
        node_feats = torch.bmm(lap, node_feats)
        node_feats = self.activation(node_feats)
        return node_feats

class GraphConv(nn.Module):
    def __init__(self, num_embeddings):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim = 64)
        self.gcn1 = GCNLayer(c_in=64, c_out=128)
        self.gcn2 = GCNLayer(c_in=128, c_out=64)

    def forward(self, node_feats, adj_matrix):
        v = self.embed(node_feats)
        v = self.gcn1(v, adj_matrix)
        v = self.gcn2(v, adj_matrix)
        v = torch.sum(v, dim=1)
        return v

class SmilesNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(40, 64)
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 64)

    def forward(self, x):
        x = self.embed(x)
        out, (h_0, c_0) = self.lstm(x, None)
        y = self.fc(out[:,-1,:])
        return y

class Classifier(nn.Sequential):
    def __init__(self, ligase_model, target_model, 
        ligase_ligand_model, target_ligand_model, 
        smiles_model):
        
        super().__init__()
        self.ligase_model = ligase_model
        self.target_model = target_model
        self.ligase_ligand_model = ligase_ligand_model
        self.target_ligand_model = target_ligand_model
        self.smiles_model = smiles_model
        self.fc1 = nn.Linear(64*5,64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64,2)

    def forward(self, ligase_atom, ligase_bond, 
        target_atom,  target_bond, 
        ligase_ligand_atom, ligase_ligand_bond, 
        target_ligand_atom, target_ligand_bond, 
        smiles):
        
        v_0 = self.ligase_model(ligase_atom, ligase_bond)
        v_1 = self.target_model(target_atom, target_bond)
        v_2 = self.ligase_ligand_model(ligase_ligand_atom, ligase_ligand_bond)
        v_3 = self.target_ligand_model(target_ligand_atom, target_ligand_bond)
        v_4 = self.smiles_model(smiles) 
        v_f = torch.cat((v_0, v_1, v_2, v_3, v_4), 1)
        v_f = self.relu(self.fc1(v_f))
        v_f = self.fc2(v_f)
        return v_f

