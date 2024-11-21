from setuptools import setup, find_packages

setup(
    name="protein_ligand_binding",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.5.0',
        'torch',
        'torch-geometric',
        'rdkit',
        'mdtraj',
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy'
    ]
)