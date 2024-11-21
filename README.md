# MMDRPv1
 
MM-DRPNet is a state-of-the-art deep learning framework for predicting protein-ligand binding affinity. By integrating multimodal features, including Dynamic Radial Partitioning (DRP) and Graph Attention Network (GAT), this model captures intricate spatial interactions and topological features, significantly improving prediction accuracy across benchmarks.

# Key Features

	•	Dynamic Radial Partitioning (DRP): Dynamically segments the 3D protein-ligand interaction space for detailed spatial feature extraction.
	•	Graph Attention Network (GAT): Captures molecular topological features to enhance the model’s contextual understanding of molecular interactions.
	•	Multimodal Framework: Fuses structural data, interaction features, and physicochemical properties for comprehensive binding affinity modeling.
	•	Superior Performance: Outperforms traditional and deep learning methods like DeepDTA, OnionNet, and AutoDock Vina in multiple benchmarks.

# Installation


## To set up the environment and dependencies, follow these steps:
1. Clone the repository:
```
git clone https://github.com/Bigrock-dd/MMDRPv1.git
cd MMDRPv1
```
2. Install dependencies via pip:
```
pip install -r requirements.txt
```

# Usage
## 1. Prepare the protein-ligand complexes (3D structures) in pdb format
```
Make sure that the protein-ligand complexes are from experimental crystals or NMR structures or molecular docking.
```
## 2. Generate dynamic radial partitioning and structure features
Within the "input.dat" file, each line provides the name and path of a protein-ligand complex. The format is as follows:
```
data/raw/11zz/11zz_complex.pdb
```
Then, you can run the command as follow to generate dynamic radial partitioning and structure features:
```
python sript.py
```
Note, the pdb file should be formatted correctly and that the ligands are recognised correctly.
## 3. Train the model
```
python scripts/train_model.py \
    -fn_train ... \
    -fn_validate ... \
    -fn_test ... \
    -batch 32 \
    -epochs 1000
```
## 4. Predicting the affinity of a protein-ligand complex
We provide the "predict.py" script in the "scoring" directory, you can score protein-ligand complexes using the following command:
```
python scripts/predict.py \
    --model ... \
    --input features.csv \
    --output predictions.csv
```
Note that you should first train the model or download our trained model (it will be uploaded on a cloud drive) and save it to the appropriate path.
