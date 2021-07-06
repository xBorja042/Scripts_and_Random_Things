f = open("C:\\..\\..\\..\\input02.txt", "r")

n_valids = 0

for line in f:
    l = line.split(' ')
    times = l[2][:-1].count(l[1][:-1])
    if times < int(l[0].split('-')[0]) or times > int(l[0].split('-')[1]):
        pass
    else: n_valids += 1
    
print(f" Solution to part one ---> {n_valids}")



f = open("C:\\..\\..\\..\\input02.txt", "r")
n_valids2 = 0
for line in f:
    l = line.split(' ')
    valid = l[1][:-1]
    if l[2][:-1][int(l[0].split('-')[0]) - 1] == valid and l[2][:-1][int(l[0].split('-')[1]) -1] == valid:
        pass
    elif l[2][:-1][int(l[0].split('-')[0]) - 1] == valid or l[2][:-1][int(l[0].split('-')[1]) -1] == valid:
        n_valids2 += 1
print(f" Solution to part two ---> {n_valids2}")
