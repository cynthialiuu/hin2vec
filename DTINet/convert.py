import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

def bipartite_to_adjacency(A):
     m, n = A.shape
     Z_mm = np.zeros((m,m), dtype=int)
     Z_nn = np.zeros((n,n), dtype=int)
     top_partition = np.concatenate((Z_nn,np.transpose(A)), axis=1)
     bottom_partition = np.concatenate((A,Z_mm), axis=1)
     return np.concatenate((top_partition, bottom_partition))

def create_matrix(fname):
    matrix = np.loadtxt(fname)
    return(matrix)

def sparse_mat(m):
    sparse_matrix = csr_matrix(m)
    return(sparse_matrix)

def convert_matrix_to_graph(m):
    g = nx.from_numpy_array(m, create_using=nx.MultiGraph)
    return(g)

def convert_bipart_to_graph(m):
    g = nx.algorithms.bipartite.matrix.from_biadjacency_matrix(m, create_using=nx.MultiGraph)
    return(g)

def write_file(g):
    with open('protein_protein_edges.txt', 'w') as f:
        for i in g.nodes:
            for j in g.neighbors(i):
                # do i need this
                if i == j:
                    continue
                elif i < 2221:
                    f.write(str(i + 1 + 708))
                    f.write("\tP\t")
                    f.write(str(j + 1 + 708))
                    f.write("\tP\t")
                    f.write("P-P")
                    f.write("\n")  

def add_missing():
    data = []

    with open('protein_protein_edges.txt', "r") as inp:  #Read phase
        data = inp.readlines()  #Reads all lines into data at the same time

    count = 709
    has_found = False
    set_num = 0
    for index, line in enumerate(data):
        all_num = line.split()
        first_num = all_num[0]
        true_num = int(first_num) 
        
        if ((has_found == True) & (true_num == set_num)):
            continue
        elif (true_num == count):
            has_found = True
            count = count + 1
            set_num = true_num
        elif (true_num != count):
            data.insert(index, str(count) + "\tP\t" + str(count) + "\tP\t" + "P-P" + "\n")  
            count = count + 1
            has_found = False

    with open('protein_protein_edges.txt', "w") as document1: #write back phase
            document1.writelines(data)



      

dmatrix = create_matrix("/Users/cindyliu/Documents/Python/PyDTINet/data/mat_protein_protein.txt")
print(len(dmatrix))
# new_m = sparse_mat(dmatrix)
dgraph = convert_matrix_to_graph(dmatrix)
print(dmatrix[0].size)
# dgraph = convert_bipart_to_graph(new_m)
write_file(dgraph)
add_missing()


