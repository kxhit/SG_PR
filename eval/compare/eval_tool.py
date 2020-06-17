import numpy as np
from scipy.spatial.distance import pdist

def f_dist(f1, f2, type="l2"):
    '''
    compute feature distance
    :param f1: feature_dim x 1
    :param f2: feature_dim x 1
    :param type: distance type
    :return:
    '''
    assert f1.shape == f2.shape
    if type == "l2":
        return np.linalg.norm(f1-f2)
    elif type=="cosine":
        return np.transpose(pdist(np.vstack([np.transpose(f1),np.transpose(f2)]), "cosine"))
    elif type == "l1":
        return np.linalg.norm(f1-f2,ord=1)
    else:
        print("this f_dist is not implemented")
        exit(-1)

def find_gt(id1,id2,gt_data):
    '''
    judge pair(id1,id2) is true or false
    id2 > id1
    :param id1: id1 in dictionary is str
    :param id2: id2 in dictionary is int
    :param gt_data: gt data
    :return: 1 - true ; 0 - false
    '''
    # assert id2 > id1
    if id2 > id1:
        id1 = str(id1)
        if id1 in gt_data:
            if id2 in gt_data[id1]:
                return 1 # (i,j) pair is in gt data
        return 0
    elif id1 == id2:
        return 1
    else:
        id2 = str(id2)
        if id2 in gt_data:
            if id1 in gt_data[id2]:
                return 1  # (i,j) pair is in gt data
        return 0