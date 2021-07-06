
tk = 'BFFFBBFRRR'

row, col = tk[:-3], tk[-3:]

def column(c:str) -> int:
    if c == 'RRR': cod = 7
    if c == 'RRL': cod = 6
    if c == 'RLR': cod = 5
    if c == 'RLL': cod = 4
    if c == 'LRR': cod = 3
    if c == 'LRL': cod = 2
    if c == 'LLL': cod = 1
    return cod

#p1, p2 = ,column(col)