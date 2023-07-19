def write_file(fname):
    with open('drug_nodes.txt', 'w') as f:
        with open('protein_nodes.txt', 'w') as g:
            with open(fname, 'r') as big:
                next(big)
                for line in big:
                    all_num = line.split()
                    first_num = all_num[0]
                    true_num = int(first_num)
                    if (true_num < 709):
                        f.write(line)
                    elif (true_num < 2221):
                        g.write(line)

write_file('node_vectors.txt')

# def add_disease():
#     with open('disease.txt', 'w') as d:
#         for i in range(1, 5604):
#             d.write(str(i + 2220) + '\tI\t' + str(i + 2220) + '\tI\t' + 'I-I\n')

# def add_se():
#     with open('se.txt', 'w') as d:
#         for i in range(1, 4193):
#             d.write(str(i + 7823) + '\tS\t' + str(i + 7823) + '\tS\t' + 'S-S\n')

# add_disease()
# add_se()
         
def check(fname):
    hello = set()
    set_copy = set()
    with open(fname, 'r') as data:
        for line in data:
            all_num = line.split()
            first_num = all_num[0]
            true_num = int(first_num)
            hello.add(true_num)
            if (hello == set_copy):
                print("False")
                print(len(hello))
                return
            set_copy.add(true_num)
        
        print(" True")
        print(len(hello))
    

check('protein_nodes.txt')




            