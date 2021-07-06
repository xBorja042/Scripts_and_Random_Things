
import os

passports = os.path.join(os.getcwd() + '\\input_04.txt')

f = open(passports, "r")


valid = sorted(['byr',
'iyr',
'eyr',
'hgt',
'hcl',
'ecl',
'pid'])

rows = [line[:-1] for line in f]

delimits = list()
n = 0
for r in rows:
    if r == '':
        delimits.append(n)
    n += 1
delimits.append(len(rows))
l_pass = list()

q = 0
for d in delimits:
    pas = list()
    for i in range(q, d):
        pas.append(rows[i])
    l_pass.append(pas)
    q = d + 1

n_valids = 0
valids = list()
for ps in l_pass:
    lk = list()
    for seps in ps:
        sub = seps.split(' ')
        for item in sub:
            kv = item.split(':')
            lk.append(kv[0])
    if 'cid' in lk: lk.remove('cid')
#    print(ps, lk)
    if sorted(list(set(lk))) == valid:
#        print(f" This is valid, {ps}")
        n_valids += 1
    valids.append(ps)
        
print(f" Solution to day 4, part one, total passports valid is ---> {n_valids}")

is_valid = list()
for ps in valids:
    if len(ps) == 0:
        ps = ps[0].split(' ')
    for item in ps:
        if 'byr' in item:
            year = int(item.split(':')[1])
            print(year)
            if year >= 1920 and year <= 2002:
                is_valid.append(True)
        if 'iyr' in item:
            year = int(item.split(':')[1])
            print(year)
            if year >= 2010 and year <= 2020:
                is_valid.append(True)
        if 'eyr' in item:
            year = int(item.split(':')[1])
            print(year)
            if year >= 2010 and year <= 2030:
                is_valid.append(True)
        if 'hgt' in item:
            height = item.split(':')[1]
            height1 = int(height[:-2])
            print(heigh1)
            print(height)
            if 'cm' in height:
                if height1 >= 150 and height1 <= 193:
                    is_valid.append(True)
            if 'in' in height:
                if height1 >= 59 and height1 <= 76:
                    is_valid.append(True)
        