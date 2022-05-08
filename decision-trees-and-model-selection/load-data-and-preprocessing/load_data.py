filename = 'sample.txt'
entries = []
# Take a provided filename and open it, calling it 'f'
with open(filename,'r') as f :
    # Read each line
    for x in f.readlines() :
        entries.append(x)
print(entries)

# To remove the trailing newline characters ('\n'),
# you can use the rstrip() method before appending:

filename = 'sample.txt'
entries = []
# Take a provided filename and open it, calling it 'f'
with open(filename,'r') as f :
    # Read each line
    for x in f.readlines() :
        entries.append(x.rstrip())
print(entries)

# An even more efficient and resilent method for constructing a
# vector is the following optimization
filename = 'sample.txt'
# Take a provided filename and open it, calling it 'f'
with open(filename,'r') as f :
    entries = [x.rstrip() for x in f.readlines() if len(x) > 0]
print(entries)