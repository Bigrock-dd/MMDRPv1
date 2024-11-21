import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
try:
    from rdkit.Chem import rdDecomposition
except ImportError:
    from rdkit.Chem.BRICS import BRICSDecompose 
import numpy as np
import warnings
import os
from pathlib import Path


class GAT(torch.nn.Module):
    
    def __init__(self, num_node_features, hidden_channels, max_atoms=1000, num_heads=8):
        super(GAT, self).__init__()
        self.max_atoms = max_atoms
        self.hidden_channels = hidden_channels
        

        self.input_transform = torch.nn.Linear(num_node_features, hidden_channels)
        

        self.gat1 = GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads)
        self.gat2 = GATConv(hidden_channels, hidden_channels, heads=1)
        self.gat3 = GATConv(hidden_channels, hidden_channels, heads=1)
        

        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        

        self.norm1 = torch.nn.LayerNorm(hidden_channels)
        self.norm2 = torch.nn.LayerNorm(hidden_channels)
        self.norm3 = torch.nn.LayerNorm(hidden_channels)
        
    def forward(self, x, edge_index, batch):

        x = self.input_transform(x)  # [num_nodes, num_node_features] -> [num_nodes, hidden_channels]
        
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

def extract_gnn_features_from_pdb(pdb_file, lig_code='LIG'):

    try:
        mol = sanitize_pdb(pdb_file, lig_code)
        if mol is None:
            raise ValueError("Failed to read and sanitize PDB file")
            
        print(f"Molecule has {mol.GetNumAtoms()} atoms")
        
        fragments = split_molecule(mol)
        if not fragments:
            raise ValueError("Failed to process molecule")
            
        print(f"Processing {len(fragments)} molecular fragments...")
        
        fragments = [f for f in fragments if is_valid_molecule(f)]
        if not fragments:
            raise ValueError("No valid fragments found")
            
        all_features = []
        model = GAT(
            num_node_features=19,
            hidden_channels=64
        )
        model.eval()
        
        for i, frag in enumerate(fragments):
            try:
                print(f"Processing fragment {i+1}/{len(fragments)}...")
                graph = mol_to_graph(frag)
                if graph is None or graph.num_nodes == 0:
                    print(f"Skipping invalid fragment {i}")
                    continue
                    
                print(f"Graph node features shape: {graph.x.shape}")
                print(f"Edge index shape: {graph.edge_index.shape}")
                
                with torch.no_grad():
                    batch = torch.zeros(graph.num_nodes, dtype=torch.long)
                    features = model(graph.x, graph.edge_index, batch)
                    print(f"Output features shape: {features.shape}")
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
        print(f"Error in extract_gnn_features_from_pdb: {e}")
        return None

def is_valid_molecule(mol):

    if mol is None:
        return False
    try:
        return (mol.GetNumAtoms() > 0 and 
                all(atom.GetAtomicNum() > 0 for atom in mol.GetAtoms()))
    except:
        return False

def get_atom_features(atom):

    if atom is None:
        return [0] * 19
        
    features = []
    
    atom_types = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I']
    atom_type = [1 if atom.GetSymbol() == t else 0 for t in atom_types]
    features.extend(atom_type)
    
    features.extend([
        atom.GetDegree(),          
        atom.GetTotalNumHs(),      
        atom.GetFormalCharge(),    
        atom.GetIsAromatic() * 1,  
        atom.GetAtomicNum(),       
        atom.GetMass(),            
        atom.GetExplicitValence(), 
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

def sanitize_pdb(pdb_file, lig_code):

    try:
        print("Reading PDB file...")
        mol = Chem.MolFromPDBFile(pdb_file, sanitize=False, removeHs=False)
        if mol is None:
            print("Failed to read PDB file")
            return None
            
        print(f"Initial molecule has {mol.GetNumAtoms()} atoms")
        

        lig_atoms = []
        for atom in mol.GetAtoms():
            residue = atom.GetPDBResidueInfo()
            if residue and residue.GetResidueName().strip() == lig_code:
                lig_atoms.append(atom.GetIdx())
                
        if not lig_atoms:
            print(f"No ligand atoms found with code {lig_code}")
            return None
            
        print(f"Found {len(lig_atoms)} ligand atoms")
        

        lig_mol = Chem.RWMol()
        atom_map = {}  
        
        for i, idx in enumerate(lig_atoms):
            atom = mol.GetAtomWithIdx(idx)
            new_atom = Chem.Atom(atom.GetAtomicNum())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
            new_atom.SetIsAromatic(atom.GetIsAromatic())
            atom_map[idx] = i
            lig_mol.AddAtom(new_atom)
            

        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            if begin_idx in lig_atoms and end_idx in lig_atoms:
                lig_mol.AddBond(atom_map[begin_idx], 
                              atom_map[end_idx],
                              bond.GetBondType())
                
        try:
            mol = lig_mol.GetMol()
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_ADJUSTHS)
            mol = Chem.RemoveHs(mol)
            if not is_valid_molecule(mol):
                raise ValueError("Invalid molecule after sanitization")
            return mol
        except Exception as e:
            print(f"Warning: Initial sanitization failed: {e}")
            try:
                mol = Chem.AddHs(lig_mol.GetMol())
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
                mol = Chem.RemoveHs(mol)
                if not is_valid_molecule(mol):
                    raise ValueError("Invalid molecule after 3D conformation")
                return mol
            except Exception as e2:
                print(f"Error generating 3D conformation: {e2}")
                return None
                
    except Exception as e:
        print(f"Error in sanitize_pdb: {e}")
        return None

def split_molecule(mol):

    if mol is None:
        return []
        
    fragments = []
    try:
        fragments = list(rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False))
    except:
        print("Warning: Failed to split molecule using GetMolFrags")
        pass
        
    if not fragments:
        try:

            try:
                decomp = list(rdDecomposition.BRICSDecompose(mol))
            except NameError:
                decomp = list(BRICSDecompose(mol))  # 使用新导入的函数
                
            for frag in decomp:
                frag_mol = Chem.MolFromSmiles(frag)
                if frag_mol is not None and frag_mol.GetNumAtoms() > 0:
                    fragments.append(frag_mol)
        except Exception as e:
            print(f"Warning: Failed to split molecule using BRICS: {e}")
            pass
            

    if not fragments and mol is not None:
        fragments = [mol]
        

    valid_fragments = []
    for frag in fragments:
        try:
            if is_valid_molecule(frag):
                valid_fragments.append(frag)
        except:
            continue
            
    return valid_fragments

def mol_to_graph(mol):

    if not is_valid_molecule(mol):
        return None
        
    try:
 
        node_features = []
        for atom in mol.GetAtoms():
            atom_features = get_atom_features(atom)
            node_features.append(atom_features)
            

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
        print(f"Error in mol_to_graph: {e}")
        return None

def extract_gnn_features_from_pdb(pdb_file, lig_code='LIG'):
    try:
        mol = sanitize_pdb(pdb_file, lig_code)
        if mol is None:
            raise ValueError("Failed to read and sanitize PDB file")
            
        print(f"Molecule has {mol.GetNumAtoms()} atoms")
        
        fragments = split_molecule(mol)
        if not fragments:
            raise ValueError("Failed to process molecule")
            
        print(f"Processing {len(fragments)} molecular fragments...")
        
        fragments = [f for f in fragments if is_valid_molecule(f)]
        if not fragments:
            raise ValueError("No valid fragments found")
            
        all_features = []
        model = GAT(
            num_node_features=19,
            hidden_channels=64
        )
        model.eval()
        
        for i, frag in enumerate(fragments):
            try:
                print(f"Processing fragment {i+1}/{len(fragments)}...")
                graph = mol_to_graph(frag)
                if graph is None or graph.num_nodes == 0:
                    print(f"Skipping invalid fragment {i}")
                    continue
                    
                print(f"Graph node features shape: {graph.x.shape}")
                print(f"Edge index shape: {graph.edge_index.shape}")
                
                with torch.no_grad():
                    batch = torch.zeros(graph.num_nodes, dtype=torch.long)
                    features = model(graph.x, graph.edge_index, batch)
                    print(f"Output features shape: {features.shape}")
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
        print(f"Error in extract_gnn_features_from_pdb: {e}")
        return None

def extract_gnn_features(smiles_list):
    try:
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
            
        print(f"Processing {len(smiles_list)} molecules...")
        
        gnn_features = []
        model = GAT(
            num_node_features=19,  
            hidden_channels=64
        )
        model.eval()
        
        for i, smiles in enumerate(smiles_list):
            try:
                print(f"Processing molecule {i+1}/{len(smiles_list)}...")
                mol = Chem.MolFromSmiles(smiles)
                if not is_valid_molecule(mol):
                    print(f"Invalid molecule from SMILES: {smiles}")
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

def main():
    import argparse
    from argparse import RawDescriptionHelpFormatter
    import pandas as pd
    from src.features.shell_features import AtomTypeCounts
    from src.features.utils import get_elementtype, normalize_features
    import itertools
    
    parser = argparse.ArgumentParser(
        description="Protein-ligand binding affinity prediction - Feature Generation",
        formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument("-inp", type=str, default="input.dat",
                        help="Input PDB file list")
    parser.add_argument("-out", type=str, default="features.csv",
                        help="Output features file")
    parser.add_argument("-lig", type=str, default="LIG",
                        help="Ligand molecule residue name")
    parser.add_argument("-clusters", type=int, default=5,
                        help="Number of dynamic shells")
    
    args = parser.parse_args()
    
    warnings.filterwarnings("ignore", category=UserWarning)
    
    print("Starting feature generation...")
    print(f"Input file: {args.inp}")
    print(f"Output file: {args.out}")
    print(f"Ligand code: {args.lig}")
    print(f"Number of clusters: {args.clusters}")
    
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    

    with open(args.inp) as f:
        pdb_files = [line.strip() for line in f if line.strip()]
    print(f"Found {len(pdb_files)} PDB files to process")
    
    ALL_ELEMENTS = ["H", "C", "O", "N", "P", "S", "HAX", "DU"]
    keys = ["_".join(x) for x in list(itertools.product(ALL_ELEMENTS, ALL_ELEMENTS))]
    
    first_time = True
    successful_count = 0
    failed_files = []
    
    for i, pdb_file in enumerate(pdb_files, 1):
        print(f"\nProcessing {i}/{len(pdb_files)}: {pdb_file}")
        
        try:
            if not os.path.exists(pdb_file):
                possible_paths = [
                    pdb_file,
                    os.path.join("data/raw", pdb_file),
                    os.path.join("data/raw", os.path.basename(pdb_file))
                ]
                found = False
                for path in possible_paths:
                    if os.path.exists(path):
                        pdb_file = path
                        found = True
                        break
                if not found:
                    raise FileNotFoundError(f"Could not find file: {pdb_file}")
            
            print("Extracting shell features...")
            cplx = AtomTypeCounts(pdb_file, args.lig)
            cplx.parsePDB(rec_sele="protein", lig_sele=args.lig)
            
            new_lig = list(map(get_elementtype, cplx.lig_ele))
            new_rec = list(map(get_elementtype, cplx.rec_ele))
            
            if not new_lig or not new_rec:
                raise ValueError("No ligand or receptor atoms found")
                
            rec_lig_element_combines = ["_".join(x) for x in list(itertools.product(new_rec, new_lig))]
            cplx.distance_pairs()
            
            dynamic_shell_features = cplx.get_dynamic_shell_features(n_clusters=args.clusters)
            
            results = []
            for count in dynamic_shell_features:
                d = {}
                d = d.fromkeys(keys, 0.0)
                for e_e, c in zip(rec_lig_element_combines, [count] * len(rec_lig_element_combines)):
                    d[e_e] = d.get(e_e, 0.0) + c
                results += list(d.values())
                
            shell_features = np.array(results).reshape(1, -1)

            print("Extracting GNN features...")
            gnn_features = extract_gnn_features_from_pdb(pdb_file, args.lig)
            
            if shell_features is None or gnn_features is None:
                raise ValueError("Failed to extract features")
                

            shell_features = normalize_features(shell_features)
            combined_features = np.concatenate([shell_features, gnn_features], axis=1)
            

            if first_time:
                num_shell_features = shell_features.shape[1]
                num_gnn_features = gnn_features.shape[1]
                shell_feature_names = [f"shell_feat_{i}" for i in range(num_shell_features)]
                gnn_feature_names = [f"GNN_feat_{i}" for i in range(num_gnn_features)]
                feature_names = shell_feature_names + gnn_feature_names
                
                with open(args.out, 'w') as f:
                    f.write("compound_id," + ",".join(feature_names) + "\n")
                first_time = False
                
            df = pd.DataFrame(combined_features, columns=feature_names)
            df.insert(0, "compound_id", pdb_file)
            df.to_csv(args.out, mode='a', header=False, index=False)
            
            successful_count += 1
            print(f"Successfully processed {pdb_file}")
            
        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")
            failed_files.append(pdb_file)
            continue
    
    print("\nFeature generation completed!")
    print(f"Successfully processed {successful_count} out of {len(pdb_files)} files")
    if failed_files:
        print("\nFailed files:")
        for f in failed_files:
            print(f"- {f}")
    print(f"Results saved to: {args.out}")

if __name__ == "__main__":
    main()
