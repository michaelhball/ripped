import numpy as np

from pathlib import Path
from tqdm import tqdm


def load_glove(path):
    we = {}
    with Path(path).open('r') as f:
        for line in tqdm(f, total=2196018):
            l = line.split(' ')
            we[l[0]] = np.asarray(l[1:], dtype=np.float32)
    
    return we
