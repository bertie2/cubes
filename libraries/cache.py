import os
import numpy as np

cache_path_fstring = "cubes_{0}.npy"

def cache_exists(n):
    cache_path = cache_path_fstring.format(n)
    return os.path.exists(cache_path)

def get_cache_raw(cache_path):
    if os.path.exists(cache_path):
        
        polycubes = np.load(cache_path, allow_pickle=True)
        
        return polycubes
    else:
        return None

def get_cache(n):
    # Check cache
    cache_path = cache_path_fstring.format(n)
    print(f"\rLoading polycubes n={n} from cache: ", end = "")
    polycubes = get_cache_raw(cache_path)
    print(f"{len(polycubes)} shapes")
    return polycubes

def save_cache_raw(cache_path, polycubes):
    np.save(cache_path, np.array(polycubes, dtype=object), allow_pickle=True)

def save_cache(n, polycubes):
    cache_path = cache_path_fstring.format(n)
    save_cache_raw(cache_path, polycubes)
    print(f"Wrote file for polycubes n={n}")