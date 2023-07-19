import numpy as np

def create_matrix(fname):
    matrix = np.loadtxt(fname)
    return(matrix)

#first turn drug_nodes and protein_nodes into easily accessable things
#then according to D-P matrix, combine vectors and find label = 1

def make_dict(fname, size, protein):
    big_v = [[]] * (size + 1)
    with open(fname, 'r') as f:
        for line in f:
            all_num = line.split()
            first_num = all_num[0]
            index = int(first_num)
            all_num.pop(0)
            converted_arr = np.array(all_num)
            converted_arr = converted_arr.astype('double')
            if (protein):
                index = index - 708
            big_v[index] = converted_arr
    return(big_v)

def find_ones(matrix, drug, protein, vector_size):
    ones_count = 0
    zero_count = 0
    ones = np.zeros((1923,vector_size*2))
    all = np.zeros((1070496,vector_size*2))
    # all = [[[] * 512] * 1512] * 708
    zeros = np.zeros((1068573,vector_size*2))
    num_rows = len(matrix)
    row_size = len(matrix[0])
    k = 0
    for i in range(0, num_rows):
        for j in range(0, row_size):
            # For testing his vectors
            # pro = protein[j]
            # combined = np.append(pro, drug[i])
            # all[k]= combined
            # k = k + 1

            pro = protein[j + 1]
            combined = np.append(pro, drug[i + 1])
            all[k]= combined
            k = k + 1
            
            
            if (matrix[i][j] == 1):
                ones[ones_count] = combined
                ones_count = ones_count + 1
            else:
                zeros[zero_count] = combined
                zero_count = zero_count + 1
    
    # return(all, ones, zeros)
    print(ones_count, zero_count)
    return(all, ones, zeros)


def make_first():
    # big = create_matrix("/Users/cindyliu/Documents/Python/hin2vec/res/mat_drug_protein.txt")
    # drug = create_matrix("/Users/cindyliu/Documents/Python/hin2vec/drug_vector_d100_dti.txt")
    # protein = create_matrix('/Users/cindyliu/Documents/Python/hin2vec/protein_vector_d100_dti.txt')

    # big = create_matrix("/home/data1/liu_cindy/hin2vec/res/mat_drug_protein.txt")
    # drug = create_matrix("/home/data1/liu_cindy/hin2vec/drug_vector_d100_dti.txt")
    # protein = create_matrix('/home/data1/liu_cindy/hin2vec/protein_vector_d100_dti.txt')

    # big = create_matrix("/Users/cindyliu/Documents/Python/hin2vec/mat_drug_protein.txt")
    # drug = make_dict("/Users/cindyliu/Documents/Python/hin2vec/drug_nodes.txt", 708, False)
    # protein = make_dict('/Users/cindyliu/Documents/Python/hin2vec/protein_nodes.txt', 1512, True)
    # ones = actual_find_ones(big, drug, protein, 100)

    big = create_matrix("/home/data1/liu_cindy/hin2vec/mat_drug_protein.txt")
    drug = make_dict("/home/data1/liu_cindy/hin2vec/drug_nodes.txt", 708, False)
    protein = make_dict('/home/data1/liu_cindy/hin2vec/protein_nodes.txt', 1512, True)

    all, ones, zeros = find_ones(big, drug, protein, 100)
    np.savetxt('all_mine.txt', all)
    np.savetxt('ones.txt', ones)
    np.savetxt('zeros.txt', zeros)

make_first()




