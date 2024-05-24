import numpy as np

def ind2sub(array_shape, ind):
    # array_shape is array,an array of two elements where the first element is the number of rows in the matrix
    # and the second element is the number of columns.
    # ind is vector index with the python rule
    ind = np.array(ind)
    array_shape = np.array(array_shape)
    rows = (ind.astype('int') // array_shape[1].astype('int'))
    cols = (ind.astype('int') % array_shape[1])  # or numpy.mod(ind.astype('int'), array_shape[1])
    return rows, cols


