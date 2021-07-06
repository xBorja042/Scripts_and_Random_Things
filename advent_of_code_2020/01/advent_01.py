f = open("C:\\...\\...\\...\\input01.txt", "r")
numbers = [int(row) for row in f]
l_goods = list()

for num1 in numbers:
    for num2 in numbers:
        suma = num1 + num2
        if suma == 2020: 
            l_goods.append(num1)
            l_goods.append(num2)
            
from numpy import prod
print(f" Solution to part one ---> {prod(list(set(l_goods)))}")
    

l_goods2 = list()
for num1 in numbers:
    for num2 in numbers:
        for num3 in numbers:
            suma = num1 + num2 + num3
            if suma == 2020: 
                l_goods2.append(num1)
                l_goods2.append(num2)
                l_goods2.append(num3)
            
print(f" Solution to part two ---> {prod(list(set(l_goods2)))}")