import numpy as np
import matplotlib.pyplot as plt
from operator import truediv
import scipy.io as sio
import torch
import math
from Utils import extract_samll_cubic
import torch.utils.data as Data
import hdf5storage

def distance(whole_data, gt_hsi,):
    indian_data_features=np.reshape(whole_data,(whole_data.shape[0]*whole_data.shape[1],whole_data.shape[2]))
    index_all=np.array(range(whole_data.shape[0]*whole_data.shape[1]))
    #gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]), )
    dist1 = compute_dist(np.array(indian_data_features), np.array(indian_data_features))
    spatial_coordinates = sptial_neighbor_matrix(index_all, gt_hsi)
    dist_spa = compute_dist(np.array(spatial_coordinates), np.array(spatial_coordinates))
    dist_spa = dist_spa / np.tile(np.sqrt(np.sum(dist_spa ** 2, 1))[..., np.newaxis], (1, dist_spa.shape[1]))
    dist = dist1 + 30 * dist_spa
    lam = 0.5 #越大，连边越少
    adj = np.zeros((np.array(indian_data_features).shape[0], np.array(indian_data_features).shape[0]))
    for k in range(np.array(indian_data_features).shape[0]):
        idxa0 = range(np.array(indian_data_features).shape[0])
        di = dist[k,:]
        ad = -0.5 * lam * di
        adj[k][idxa0] = EProjSimplex_new(ad)
    #adj = coo_matrix(adj)
    #adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def sptial_neighbor_matrix(index_all, gt):
    """ extract the spatial coordinates(x,y) """

    L_cor = np.zeros([2, 1])
    for kkk in range(len(index_all)):
        [X_cor, Y_cor] = ind2sub([np.size(gt, 0), np.size(gt, 1)], index_all[kkk])
        XY_cor = np.array([X_cor, Y_cor])[..., np.newaxis]
        L_cor = np.concatenate((L_cor, XY_cor), axis=1)
    L_cor = np.delete(L_cor, 0, axis=1)  # delete 0 column

    return L_cor.transpose()




def compute_dist(array1, array2, type='euclidean'):
    """Compute the euclidean or cosine distance of all pairs.
  Args:
    array1: numpy array with shape [m1, n]
    array2: numpy array with shape [m2, n]
    type: one of ['cosine', 'euclidean']
  Returns:
    numpy array with shape [m1, m2]
  """

    assert type in ['cosine', 'euclidean']
    if type == 'cosine':
        #array1 = normalize(array1, axis=1)
       # array2 = normalize(array2, axis=1)
        dist = np.matmul(array1, array2.T)
        return dist
    else:
        # shape [m1, 1]
        square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
        # shape [1, m2]
        square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
        squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
        squared_dist[squared_dist < 0] = 0
        dist = np.sqrt(squared_dist)#np.exp(dist - np.tile(np.max(dist, axis=0)[..., np.newaxis], np.size(dist, 1)))
        return dist



#def normalize(nparray, order=2, axis=0):
#    """Normalize a N-D numpy array along the specified axis."""
#    nparray = np.array(nparray)
#    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
#    return nparray / (norm + np.finfo(np.float32).eps)

def ind2sub(array_shape, ind):
    # array_shape is array,an array of two elements where the first element is the number of rows in the matrix
    # and the second element is the number of columns.
    # ind is vector index with the python rule
    ind = np.array(ind)
    array_shape = np.array(array_shape)
    rows = (ind.astype('int') // array_shape[1].astype('int'))
    cols = (ind.astype('int') % array_shape[1])  # or numpy.mod(ind.astype('int'), array_shape[1])
    return rows, cols

def EProjSimplex_new(v, k=1):
    v = np.matrix(v)
    ft = 1;
    n = np.max(v.shape)

    #    if len(v.A[0]) == 0:
    #        return v, ft

    if np.min(v.shape) == 0:
        return v, ft

    #    print('n : ', n)
    #    print(v.shape)
    #    print('v :  ', v)

    v0 = v - np.mean(v) + k / n

    #    print('v0 :  ', v0)

    vmin = np.min(v0)

    if vmin < 0:
        f = 1
        lambda_m = 0
        while np.abs(f) > 10 ** -10:
            v1 = v0 - lambda_m
            posidx = v1 > 0
            npos = np.sum(posidx)
            g = -npos
            f = np.sum(v1[posidx]) - k
            lambda_m = lambda_m - f / g
            ft = ft + 1
            if ft > 100:
                v1[v1 < 0] = 0.0
                break
        x = v1.copy()
        x[x < 0] = 0.0
    else:
        x = v0
    return x