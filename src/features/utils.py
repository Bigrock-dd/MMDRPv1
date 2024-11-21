import numpy as np

def get_elementtype(e):

    ALL_ELEMENTS = ["H", "C", "O", "N", "P", "S", "HAX", "DU"]
    
    if e in ALL_ELEMENTS:
        return e
    elif e in ['Cl', 'Br', 'I', 'F']:
        return 'HAX'
    else:
        return "DU"

def normalize_features(features):

    min_val = np.min(features, axis=1, keepdims=True)
    max_val = np.max(features, axis=1, keepdims=True)
    return (features - min_val) / (max_val - min_val + 1e-8)