import numpy as np
import h5py

import IPython
import numpy as np
import os   #Use for playing sound.
from IPython.display import Audio
from IPython.display import display as d


def save_dict_to_hdf5(dic, filename):
    """
    ....
    """
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))

def load_dict_from_hdf5(filename):
    """
    ....
    """
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

def IsRunOnColab():
    try:
        import google.colab
        return True
    except:
        return False

def playSoundFinish():
    if IsRunOnColab():
        playSoundOnColab()
    else:
        duration = 2  # second
        freq = 500  # Hz
        os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))

def playSoundOnColab():
    print("Colab plays sound")
    # Create a sound
    framerate = 44100
    t = np.linspace(0,5,framerate*5)
    data = np.sin(2*np.pi*220*t) + np.sin(2*np.pi*224*t)

    # Generate a player for mono sound
    d(Audio(data,rate=framerate, autoplay=True))


