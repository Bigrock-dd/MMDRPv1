import numpy as np
import mdtraj as mt
import itertools
from collections import OrderedDict
from sklearn.cluster import KMeans

ALL_ELEMENTS = ["H", "C", "O", "N", "P", "S", "HAX", "DU"]

class AtomTypeCounts(object):
    """Featurization of Protein-Ligand Complex based on dynamic distance-based counts of atom-types."""

    def __init__(self, pdb_fn, lig_code):
        self.pdb = mt.load_pdb(pdb_fn)
        self.receptor_indices = np.array([])
        self.ligand_indices = np.array([])
        self.rec_ele = np.array([])
        self.lig_ele = np.array([])
        self.lig_code = lig_code
        self.pdb_parsed_ = False
        self.distance_computed_ = False
        self.distance_matrix_ = np.array([])
        self.counts_ = np.array([])

    def parsePDB(self, rec_sele="protein", lig_sele="UNK"):
        top = self.pdb.topology
        self.receptor_indices = top.select(rec_sele)
        self.ligand_indices = top.select("resname " + lig_sele)
        table, bond = top.to_dataframe()
        self.rec_ele = table['element'][self.receptor_indices]
        self.lig_ele = table['element'][self.ligand_indices]
        self.pdb_parsed_ = True
        return self

    def distance_pairs(self):
        if not self.pdb_parsed_:
            self.parsePDB()


        all_pairs = list(itertools.product(self.receptor_indices, self.ligand_indices))
        all_pairs = np.array(all_pairs)

        if not self.distance_computed_:
            self.distance_matrix_ = mt.compute_distances(self.pdb, atom_pairs=all_pairs)[0]

        self.distance_computed_ = True
        return self

    def dynamic_shell_clustering(self, n_clusters=5):
        if not self.distance_computed_:
            self.distance_pairs()


        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        distance_matrix_reshaped = self.distance_matrix_.reshape(-1, 1)
        labels = kmeans.fit_predict(distance_matrix_reshaped)


        shell_counts = np.zeros((n_clusters,))
        for label in labels:
            shell_counts[label] += 1

        return shell_counts

    def get_dynamic_shell_features(self, n_clusters=5):
        shell_counts = self.dynamic_shell_clustering(n_clusters)
        return shell_counts