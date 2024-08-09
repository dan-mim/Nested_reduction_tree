import pickle
from application2 import *

name = f'outputs/T7_mpi1.pkl'
with open(name, 'rb') as f:
    res = pickle.load(f)

print(res['MAM']['record_t_n'])
print(res['LP']['record_t_n'])

T = 4
H = make_growing_tree(T, [1, 2, 10,1])
draw_tree(H)