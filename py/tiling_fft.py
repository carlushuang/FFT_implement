import numpy as np
import scipy as sp
import scipy.signal
import sys


def corr1d_out_size(i_size, f_size):
    return i_size-f_size+1

def valid_array(lhs,rhs,atol=0.00001):
    r = np.allclose(lhs,rhs,atol)
    return r

def chose_cell_size_1d(i_size, o_size, f_size):
    cell_sizes=[8,16,32,64]
    reduce_num = sys.maxsize
    opt_cell = cell_sizes[0]
    touched = False
    for c in cell_sizes:
        #print("c:{},f:{}".format(c,f_size))
        if f_size > c:
            continue
        touched = True
        dx = corr1d_out_size(c,f_size)      # every cell can compute how many output
        num_cells = (o_size+dx-1)//dx
        rn = num_cells*(c//2+1)   # assume have r2c optimize
        if rn < reduce_num:
            reduce_num = rn
            opt_cell = c
    if not touched:
        print("filter size{} not suitable for cell!".format(f_size))
    return opt_cell

def tiling_corr_1d(i_size, f_size):
    '''
    input tiling, filter not tiling
    '''
    i_array = np.random.random_sample(i_size)
    f_array = np.random.random_sample(f_size)
    o_size = corr1d_out_size(i_size, f_size)
    o_array = np.correlate(i_array, f_array)    # reference output
    assert len(o_array) == o_size
    cell_size = chose_cell_size_1d(i_size, o_size, f_size)

    dx =  corr1d_out_size(cell_size,f_size)
    num_cells = (o_size+dx-1)//dx

    # input cells number are same as output cells, but can not computed as below, for there are overlaps
    #num_in_cells = (i_size+dx-1)//dx
    #assert num_cells == num_in_cells

    o_array_tiling = np.array([])
    #print("ref:{}".format(o_array))
    for i in range(num_cells):
        # actuall data size in current input cell
        in_data_size = cell_size if cell_size < (i_size-i*dx) else (i_size-i*dx)
        in_start = i*dx
        in_data = i_array[in_start:in_start+in_data_size]   # split is [), right exclusive
        o_tiling = np.correlate(in_data, f_array)
        o_array_tiling = np.append(o_array_tiling,o_tiling)
        #print("{},i:{}:{}".format(i,in_data_size,o_tiling))

    print("i_size:{}, f_size:{}, o_size:{}, cell_size:{}, ncells:{}, incells:{}. dx:{}".format(
            i_size, f_size, o_size, cell_size, num_cells, num_in_cells, dx))
    result = valid_array(o_array,o_array_tiling)
    if not result:
        print(o_array)
        print(o_array_tiling)

def tiling_corr_1d_due(i_size, f_size):
    '''
    both input and filter need tiling
    '''
    


if __name__ == '__main__':
    i_size = 17
    f_size = 3
    if len(sys.argv) > 1:
        i_size = int(sys.argv[1])
        f_size = int(sys.argv[2])
    tiling_corr_1d(i_size, f_size)
