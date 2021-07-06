
import os

topology = os.path.join(os.getcwd() + '\\input_03.txt')
#print(topology)

                        

f = open(topology, "r")

x0, y0 = 0, 0

rows = [line[:-1] for line in f]

n_trees = 0
for row in rows[1:]:
    x0 += 3
    if x0 >= len(row):
#        print(" We have gone too far right, break")
        x0 = 0 + x0 - len(row)
#    print(x0)
    pos = row[x0]
    if pos == "#": n_trees += 1
    
    
print(f"Solution of total trees PART ONE is ---> {n_trees}")

l_steps = [1,3,5,7]

#l_steps = [7]
ln_trees = list()
for nsteps in l_steps:
    print(nsteps)
    n_trees = 0
    x0 = 0
    for row in rows[1:]:
        print(x0)
#        if x0 >= len(row):
#            x0 = len(row)
#        else:
#            x0 += nsteps
        x0 += nsteps
        if x0 >= len(row):
#            print(" We have gone too far right, break")
            print(" lasT ", x0, len(row))
            x0 = -1 + x0 - len(row)
#        print(x0)
        pos = row[x0]
        if pos == "#": n_trees += 1
    print(x0)
    print(n_trees)
    ln_trees.append(n_trees)

#n_trees = 0  
#x0 = 0
#for i in range(1, len(rows), 2):
#    x0 += 1
#    if x0 >= len(row):
##        print(" We have gone too far right, break")
#        x0 = 0 + x0 - len(row)
##    print(x0)
#    pos = row[x0]
#    if pos == "#": n_trees += 1
#    
#ln_trees.append(n_trees)
        
import numpy
print(f"Solution of total trees PART TWO is ---> {numpy.prod(ln_trees)}")