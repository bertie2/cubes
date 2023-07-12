import os
import math
import numpy as np
import argparse
import multiprocessing
import matplotlib.pyplot as plt
from time import perf_counter

def all_rotations(polycube):
    """
    Calculates all rotations of a polycube.
  
    Adapted from https://stackoverflow.com/questions/33190042/how-to-calculate-all-24-rotations-of-3d-array.
    This function computes all 24 rotations around each of the axis x,y,z. It uses numpy operations to do this, to avoid unecessary copies.
    The function returns a generator, to avoid computing all rotations if they are not needed.
  
    Parameters:
    polycube (np.array): 3D Numpy byte array where 1 values indicate polycube positions

    Returns:
    generator(np.array): Yields new rotations of this cube about all axes
  
    """
    def single_axis_rotation(polycube, axes):
        """Yield four rotations of the given 3d array in the plane spanned by the given axes.
        For example, a rotation in axes (0,1) is a rotation around axis 2"""
        for i in range(4):
             yield np.rot90(polycube, i, axes)

    # 4 rotations about axis 0
    yield from single_axis_rotation(polycube, (1,2))

    # rotate 180 about axis 1, 4 rotations about axis 0
    yield from single_axis_rotation(np.rot90(polycube, 2, axes=(0,2)), (1,2))

    # rotate 90 or 270 about axis 1, 8 rotations about axis 2
    yield from single_axis_rotation(np.rot90(polycube, axes=(0,2)), (0,1))
    yield from single_axis_rotation(np.rot90(polycube, -1, axes=(0,2)), (0,1))

    # rotate about axis 2, 8 rotations about axis 1
    yield from single_axis_rotation(np.rot90(polycube, axes=(0,1)), (0,2))
    yield from single_axis_rotation(np.rot90(polycube, -1, axes=(0,1)), (0,2))

def crop_cube(cube):
    """
    Crops an np.array to have no all-zero padding around the edge.

    Given in https://stackoverflow.com/questions/39465812/how-to-crop-zero-edges-of-a-numpy-array
  
    Parameters:
    cube (np.array): 3D Numpy byte array where 1 values indicate polycube positions
  
    Returns:
    np.array: Cropped 3D Numpy byte array equivalent to cube, but with no zero padding
  
    """
    for i in range(cube.ndim):
        cube = np.swapaxes(cube, 0, i)
        nonzero_indices = np.any(cube != 0, axis=tuple(range(1, cube.ndim)))
        cube = cube[nonzero_indices]
        cube = np.swapaxes(cube, 0, i)
    return cube

def expand_cube(cube):
    """
    Expands a polycube by adding single blocks at all valid locations.
  
    Calculates all valid new positions of a polycube by shifting the existing cube +1 and -1 in each dimension.
    New valid cubes are returned via a generator function, in case they are not all needed.
  
    Parameters:
    cube (np.array): 3D Numpy byte array where 1 values indicate polycube positions
  
    Returns:
    generator(np.array): Yields new polycubes that are extensions of cube
  
    """
    cube = np.pad(cube, 1, 'constant', constant_values=0)
    output_cube = np.array(cube)

    xs,ys,zs = cube.nonzero()
    output_cube[xs+1,ys,zs] = 1
    output_cube[xs-1,ys,zs] = 1
    output_cube[xs,ys+1,zs] = 1
    output_cube[xs,ys-1,zs] = 1
    output_cube[xs,ys,zs+1] = 1
    output_cube[xs,ys,zs-1] = 1

    exp = (output_cube ^ cube).nonzero()

    for (x,y,z) in zip(exp[0], exp[1], exp[2]):
        new_cube = np.array(cube)
        new_cube[x,y,z] = 1
        yield crop_cube(new_cube)

def unpack_hashes_task(args):
    cube_hashes, logging_queue = args
    return [unpack(cube_hash) for cube_hash in cube_hashes]


def hash_cubes_task(args):
    base_cubes, logging_queue = args
    # Empty list of new n-polycubes
    polycubes = set()
    uid = os.getpid()

    n = 0
    for base_cube in base_cubes:
        for new_cube in expand_cube(base_cube):
            cube_hash = get_canoincal_packing(new_cube)
            polycubes.add(cube_hash)
        if(n%1000 == 0):
            logging_queue.put((uid, n, len(base_cubes)))
        n += 1

    return polycubes

def dispatch_tasks(task_function, items, logging_queue):
    if(True):
        cores = multiprocessing.cpu_count()
        chunk_size = math.ceil(len(items) / cores)

        chunks = []
        for chunk_base in range(0, len(items), chunk_size):
            chunks.append((items[chunk_base: min( chunk_base + chunk_size, len(items))], logging_queue))
        items = None
        pool = multiprocessing.Pool(cores)
        return pool.map(task_function, chunks)
    else:
        return [task_function(items)]

def generate_polycubes(n, use_cache=False, logging_queue=None):
    """
    Generates all polycubes of size n
  
    Generates a list of all possible configurations of n cubes, where all cubes are connected via at least one face.
    Builds each new polycube from the previous set of polycubes n-1.
    Uses an optional cache to save and load polycubes of size n-1 for efficiency.
  
    Parameters:
    n (int): The size of the polycubes to generate, e.g. all combinations of n=4 cubes.
  
    Returns:
    list(np.array): Returns a list of all polycubes of size n as numpy byte arrays
  
    """
    if n < 1:
        return []
    elif n == 1:
        return [np.ones((1,1,1), dtype=np.byte)]
    elif n == 2:
        return [np.ones((2,1,1), dtype=np.byte)]

    # Check cache
    cache_path = f"cubes_{n}.npy"
    if use_cache and os.path.exists(cache_path):
        print(f"\rLoading polycubes n={n} from cache: ", end = "")
        polycubes = np.load(cache_path, allow_pickle=True)
        print(f"{len(polycubes)} shapes")
        return polycubes

    results = dispatch_tasks(hash_cubes_task, generate_polycubes(n-1, use_cache, logging_queue), logging_queue)

    final_result = set()
    for result in results:
        final_result |= result
    final_result = list(final_result)

    print(f"Hashed polycubes n={n}")

    results = dispatch_tasks(unpack_hashes_task, final_result, logging_queue)

    final_result = []
    for result in results:
        final_result += result

    print(f"Generated polycubes n={n}")
    
    if use_cache:
        cache_path = f"cubes_{n}.npy"
        np.save(cache_path, np.array(final_result, dtype=object), allow_pickle=True)
        print(f"Wrote file for polycubes n={n}")

    return final_result

def pack(polycube: np.ndarray):
    """
    Converts a 3D ndarray into a single unsigned integer for quick hashing and efficient storage

    Converts a {0,1} nd array into a single unique large integer
  
    Parameters:
    polycube (np.array): 3D Numpy byte array where 1 values indicate polycube positions
  
    Returns:
    int: a unique integer hash

    """

    pack_cube = np.packbits(polycube.flatten(), bitorder='big')
    cube_hash = 0
    for index in polycube.shape:
        cube_hash = (cube_hash << 8) + int(index)
    for part in pack_cube:
        cube_hash = (cube_hash << 8) + int(part)
    return cube_hash

def unpack(cube_hash):
    """
    Converts a single integer back into a 3D ndarray


    Parameters:
    cube_hash (int): a unique integer hash
  
    Returns:
    np.array: 3D Numpy byte array where 1 values indicate polycube positions

    """
    parts = []
    while(cube_hash):
        parts.append(cube_hash%256)
        cube_hash >>= 8
    parts = parts[::-1]
    shape = (parts[0],parts[1],parts[2])
    data = parts[3:]
    size = shape[0] * shape[1] * shape[2]
    raw = np.unpackbits(np.array(data, dtype=np.uint8), bitorder='big')
    final =  raw[0:size].reshape(shape)
    return final


def get_canoincal_packing(polycube):
    """
    Determines if a polycube has already been seen.
  
    Considers all possible rotations of a cube against the existing cubes stored in memory.
    Returns True if the cube exists, or False if it is new.
  
    Parameters:
    polycube (np.array): 3D Numpy byte array where 1 values indicate polycube positions
  
    Returns:
    boolean: True if polycube is already present in the set of all cubes so far.
    hash: the hash for this cube
  
    """
    max_hash = 0
    for cube_rotation in all_rotations(polycube):
        this_hash = pack(cube_rotation)
        if(this_hash > max_hash):
            max_hash = this_hash
    return max_hash

# # Code for if you want to generate pictures of the sets of cubes. Will work up to about n=8, before there are simply too many!
# # Could be adapted for larger cube sizes by splitting the dataset up into separate images.
# def render_shapes(shapes, path):
#     n = len(shapes)
#     dim = max(max(a.shape) for a in shapes)
#     i = math.isqrt(n) + 1
#     voxel_dim = dim * i
#     voxel_array = np.zeros((voxel_dim + i,voxel_dim + i,dim), dtype=np.byte)
#     pad = 1
#     for idx, shape in enumerate(shapes):
#         x = (idx % i) * dim + (idx % i)
#         y = (idx // i) * dim + (idx // i)
#         xpad = x * pad
#         ypad = y * pad
#         s = shape.shape
#         voxel_array[x:x + s[0], y:y + s[1] , 0 : s[2]] = shape

#     #voxel_array = crop_cube(voxel_array)
#     colors = np.empty(voxel_array.shape, dtype=object)
#     colors[:] = '#FFD65DC0'

#     ax = plt.figure(figsize=(20,16), dpi=600).add_subplot(projection='3d')
#     ax.voxels(voxel_array, facecolors = colors, edgecolor='k', linewidth=0.1)
    
#     ax.set_xlim([0, voxel_array.shape[0]])
#     ax.set_ylim([0, voxel_array.shape[1]])
#     ax.set_zlim([0, voxel_array.shape[2]])
#     plt.axis("off")
#     ax.set_box_aspect((1, 1, voxel_array.shape[2] / voxel_array.shape[0]))
#     plt.savefig(path + ".png", bbox_inches='tight', pad_inches = 0)

def logging_task(queue):
    status = {}
    while True:
        (uid, done, total) = queue.get()
        status[uid] = (done, total)

        total_done = 0
        total_total = 0
        for uid, (done, total) in status.items():
            total_done += done
            total_total += total

        print(f'\rCompleted {total_done} of {total_total} {((total_done/total_total) * 100):.2f}%', end="")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Polycube Generator',
                    description='Generates all polycubes (combinations of cubes) of size n.')

    parser.add_argument('n', metavar='N', type=int,
                    help='The number of cubes within each polycube')
    
    #Requires python >=3.9
    parser.add_argument('--cache', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    n = args.n
    use_cache = args.cache if args.cache is not None else True

    with multiprocessing.Manager() as Manager:
        logging_queue = Manager.Queue()
        logging_process = multiprocessing.Process(target=logging_task, args=[logging_queue])
        logging_process.daemon = True
        logging_process.start()

        # Start the timer
        t1_start = perf_counter()

        all_cubes = list(generate_polycubes(n, use_cache=use_cache, logging_queue=logging_queue))

        # Stop the timer
        t1_stop = perf_counter()

        # padded = [np.pad(shape, 1, constant_values=0) for shape in all_cubes]
            
        # render_shapes(padded, "./out")

        print (f"\nFound {len(all_cubes)} unique polycubes")
        print (f"\nElapsed time: {round(t1_stop - t1_start,3)}s")
