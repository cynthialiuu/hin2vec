
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
