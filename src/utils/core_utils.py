
import pickle5 as pickle
import json
import pandas as pd
from tqdm import tqdm
from collections import Counter, defaultdict
import numpy as np
from functools import reduce
from itertools import product
import os
import matplotlib.pyplot as plt
import sys

from .constants import *

tqdm.pandas()

DATA_OUT_DIR = "/ais/hal9000/datasets/reddit/stance_pipeline/classifiers/"

class Serialization:
    """Wrapper for saving and loading serialized objects."""
    
    @staticmethod
    def save_obj(obj, name):
        """
        serialization of an object
        :param obj: object to serialize
        :param name: file name to store the object
        """
        with open(DATA_OUT_DIR + name + '.pkl', 'wb') as fout:
            pickle.dump(obj, fout, pickle.HIGHEST_PROTOCOL)


    @staticmethod
    def load_obj(name):
        """
        de-serialization of an object
        :param name: file name to load the object from
        """
        with open(DATA_OUT_DIR + name + '.pkl', 'rb') as fout:
            return pickle.load(fout)


def set_intersection(l1, l2):
    """
    Returns the intersection of two lists.
    """
    return list(set(l1).intersection(set(l2)))


def set_union(l1, l2):
    """
    Returns the union of two lists.
    """
    return list(set(l1).union(set(l2)))

def set_difference(l1, l2):
    """
    Returns the difference of two lists.
    """
    return list(set(l1).difference(set(l2)))

def intersect_overlap(l1, l2):
    """
    Returns the intersection of two lists,
    while also describing the size of each list
    and the size of their intersection.
    """
    print(len(l1))
    print(len(l2))
    intersected = set_intersection(l1, l2)
    print(len(intersected))
    return intersected

def jaccard_similarity(l1, l2):
    l1 = set(l1)
    l2 = set(l2)
    return np.round(len(l1.intersection(l2)) / len(l1.union(l2)), 2)

def flatten_logic(arr):
    """
    Flattens a nested array. 

    """
    for i in arr:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


def flatten(arr):
    """
    Wrapper for the generator returned by flatten logic.
    """
    return list(flatten_logic(arr))


def make_directory(dir_name):
    if not os.path.exists(dir_name):
        print(f"Creating directory {dir_name}")
        os.makedirs(dir_name)