import json
import numpy as np
import random

def config_setup():
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    random.seed(config['RANDOM_SEED'])
    np.random.seed(config['RANDOM_SEED'])



