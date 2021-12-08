#   Utility functions
#   Author:  Stephen Menary  (stmenary@cern.ch)

import os

import numpy as np


#  Return a copy of the dataset; if input is a dictionary then concatenate all datasets
#
def get_flat_copy_of_data (x) :
    if type(x) is np.ndarray : 
        return np.copy(x)
    if type(x) is list :
        return np.array(x)
    if type(x) is dict :
        x_tmp = None
        for key, item in x.items() :
            if type(item) != np.ndarray : 
                raise TypeError(f"Dictionary items are type {type(item)} where {np.ndarray} expected")
        return np.concatenate([item for key, item in x.items()])
    raise TypeError(f"Could not flatten object of type {type(x)}")


#  Print INFO message
#
def INFO (location, msg, max_location_length=20) :
    MESSAGE("INFO", location, msg, max_location_length=max_location_length)


#  Print message
#
def MESSAGE (preamble, location, msg, max_preamble_length=10, max_location_length=20) :
    preamble = preamble[:min([len(preamble), max_preamble_length])].ljust(max_preamble_length)
    location = location[max([0, len(location)-max_location_length]):].ljust(max_location_length)
    print(f"{preamble}  {location}  {msg}")



#  Shuffle rows of a numpy array
#
def joint_shuffle (*v) :
    pairs = [_ for _ in zip(*v)]
    np.random.shuffle(pairs)
    to_ret = []
    for idx in range(len(v)) :
        to_ret.append(np.array([p[idx] for p in pairs]))
    return tuple(to_ret)


#  Function for ensuring that a directory exist
#
def make_sure_dir_exists_for_filename (path) :
    path = path.strip()
    if len(path) == 0 : return
    components = path.split("/")
    if "." in components[-1] : components[-1] = ""
    components = [x for x in components if len(x) > 0]
    if path[0] == "/" : components[0] = "/" + components[0]
    target_path = "" 
    for i in components :
        if len(target_path) > 0 : target_path = target_path + "/"
        target_path = target_path + i
        if os.path.exists(target_path) :
            if os.path.isdir(target_path) : continue
            return False
        os.mkdir(target_path)
        