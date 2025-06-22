import random
import numpy as np
import sklearn.utils
import torch
import sklearn

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    sklearn.utils.check_random_state(seed)
