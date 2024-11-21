# src/features/gnn_features.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
import numpy as np

class GAT(torch.nn.Module):
    """Graph Attention Network for molecular feature extraction"""
    
    def __init__(self, num_node_features, hidden_channels, max_atoms=1000, num_heads=8):
        super(GAT, self).__init__()
        self.max_atoms = max_atoms
        self.hidden_channels = hidden_channels
        

        self.gat1 = GATConv(num_node_features, hidden_channels // num_heads, heads=num_heads)
        self.gat2 = GATConv(hidden_channels, hidden_channels, heads=1)
        self.gat3 = GATConv(hidden_channels, hidden_channels, heads=1)
        

        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        
        self.norm1 = torch.nn.LayerNorm(hidden_channels)
        self.norm2 = torch.nn.LayerNorm(hidden_channels)
        self.norm3 = torch.nn.LayerNorm(hidden_channels)
        
    def forward(self, x, edge_index, batch):

        if x.size(0) > self.max_atoms:

            outputs = []
            for i in range(0, x.size(0), self.max_atoms):
                end = min(i + self.max_atoms, x.size(0))
                mask = (batch >= i) & (batch < end)
                sub_x = x[mask]
                sub_edge_index = edge_index[:, mask[edge_index[0]] & mask[edge_index[1]]]
                sub_batch = batch[mask] - i
                
                sub_x = self.process_batch(sub_x, sub_edge_index, sub_batch)
                outputs.append(sub_x)
                
            x = torch.cat(outputs, dim=0)
        else:
            x = self.process_batch(x, edge_index, batch)
            
        return x
        
    def process_batch(self, x, edge_index, batch):

        identity = x
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.norm1(x)
        

        identity2 = x
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.norm2(x)
        x = x + identity2
        

        identity3 = x
        x = self.gat3(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.norm3(x)
        x = x + identity3
        
        
        x = global_mean_pool(x, batch)
        

        x = self.lin1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        
        return x

def get_atom_features(atom):

    if atom is None:
        return [0] * 16  # 默认特征向量
        
    features = []
    

    atom_types = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I']
    atom_type = [1 if atom.GetSymbol() == t else 0 for t in atom_types]
    features.extend(atom_type)
    

    features.extend([
        atom.GetDegree(),          # 度
        atom.GetTotalNumHs(),      # 氢原子数
        atom.GetFormalCharge(),    # 形式电荷
        atom.GetIsAromatic() * 1,  # 芳香性
        atom.GetAtomicNum(),       # 原子序数
    ])
    

    hyb_types = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3
    ]
    features.extend([1 if atom.GetHybridization() == t else 0 for t in hyb_types])
    
    return features

def get_bond_features(bond):

    if bond is None:
        return [0] * 7  
        
    features = []
    

    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]
    bond_type = [1 if bond.GetBondType() == t else 0 for t in bond_types]
    features.extend(bond_type)
    

    features.extend([
        bond.GetIsConjugated() * 1,  
        bond.IsInRing() * 1,         
        bond.GetIsAromatic() * 1     
    ])
    
    return features

def split_molecule(mol):
    if mol is None:
        return []
        
    fragments = []

    try:
        decomp = rdDecomposition.BRICSDecompose(mol)
        for frag in decomp:
            frag_mol = Chem.MolFromSmiles(frag)
            if frag_mol is not None and frag_mol.GetNumAtoms() > 0:
                fragments.append(frag_mol)
    except:
        pass
        

    if len(fragments) < 2:
        try:
            bonds = mol.GetBonds()
            if len(bonds) > 0:
                bond_indices = list(range(len(bonds)))

                selected_bonds = []
                for i, bond in enumerate(bonds):
                    if not bond.IsInRing() and bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                        selected_bonds.append(i)
                if selected_bonds:
                    fragments = Chem.FragmentOnBonds(mol, selected_bonds)
        except:
            pass
    

    if not fragments:
        return [mol]
    
    return fragments

def mol_to_graph(mol):
    if mol is None:
        return None
        
    try:
        node_features = []
        for atom in mol.GetAtoms():
            atom_features = get_atom_features(atom)
            node_features.append(atom_features)
            

        if not node_features:
            return None
            

        edge_indices = []
        edge_features = []
        for bond in mol.GetBonds():

            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            

            edge_indices.extend([[i, j], [j, i]])
            

            bond_features = get_bond_features(bond)
            edge_features.extend([bond_features, bond_features])
            
        if not edge_indices and node_features:
            edge_indices = [[i, i] for i in range(len(node_features))]
            edge_features = [get_bond_features(None) for _ in range(len(node_features))]
            

        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data
        
    except Exception as e:
        print(f"Error converting molecule to graph: {e}")
        return None

def extract_gnn_features_from_pdb(pdb_file, lig_code='LIG'):

    try:
        print("Reading PDB file...")
        mol = Chem.MolFromPDBFile(pdb_file, sanitize=False, removeHs=True)
        if mol is None:
            raise ValueError("Failed to read PDB file")
            

        mol = Chem.RemoveHs(mol, sanitize=False)
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_ADJUSTHS)
        except:
            print("Warning: Failed to sanitize molecule")
            

        fragments = split_molecule(mol)
        if not fragments:
            raise ValueError("Failed to process molecule")
            
        print(f"Processing {len(fragments)} molecular fragments...")
        
        all_features = []
        model = GAT(
            num_node_features=16,   # 匹配get_atom_features的输出维度
            hidden_channels=64,     # 隐藏层维度
            max_atoms=1000         # 最大原子数限制
        )
        model.eval()
        
        for i, frag in enumerate(fragments):
            try:
                print(f"Processing fragment {i+1}/{len(fragments)}...")
                graph = mol_to_graph(frag)
                if graph is None:
                    continue
                    
                with torch.no_grad():
                    batch = torch.zeros(graph.num_nodes, dtype=torch.long)
                    features = model(graph.x, graph.edge_index, batch)
                    all_features.append(features.numpy())
            except Exception as e:
                print(f"Warning: Failed to process fragment {i}: {e}")
                continue
                
        if not all_features:
            raise ValueError("Failed to extract features from any fragment")
            

        combined_features = np.mean(all_features, axis=0)
        if len(combined_features.shape) == 1:
            combined_features = combined_features.reshape(1, -1)
            
        return combined_features
        
    except Exception as e:
        print(f"Error extracting GNN features: {e}")
        return None

def extract_gnn_features(smiles_list):
    try:
        # 处理输入SMILES
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
            
        print(f"Processing {len(smiles_list)} molecules...")
        
        gnn_features = []
        model = GAT(
            num_node_features=16,
            hidden_channels=64
        )
        model.eval()
        
        for i, smiles in enumerate(smiles_list):
            try:
                print(f"Processing molecule {i+1}/{len(smiles_list)}...")
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    print(f"Failed to parse SMILES: {smiles}")
                    continue
                    
 
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
                mol = Chem.RemoveHs(mol)
                
                graph = mol_to_graph(mol)
                if graph is None:
                    continue
                    
                with torch.no_grad():
                    batch = torch.zeros(graph.num_nodes, dtype=torch.long)
                    features = model(graph.x, graph.edge_index, batch)
                    gnn_features.append(features.numpy())
                    
            except Exception as e:
                print(f"Error processing SMILES {smiles}: {e}")
                continue
                
        if not gnn_features:
            return None
            

        features = np.array(gnn_features)
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
            
        return features
        
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None

if __name__ == "__main__":

    test_smiles = "CC(=O)O"
    features = extract_gnn_features(test_smiles)
    if features is not None:
        print("Features shape:", features.shape)
        print("Features:", features)