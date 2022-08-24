def boxer(boks, char):
    new_boks = [char*(len(boks)+2)]
    for row in boks:
        new_boks.append(char+row+char)
    new_boks.append(char*(len(boks)+2))
    return new_boks


ab_box = boxer(['a'], 'b')
abc_box = boxer(ab_box, 'c')

print(ab_box)
print(abc_box)
