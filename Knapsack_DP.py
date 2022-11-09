import numpy as np
from pprint import pprint

products = np.genfromtxt("products.csv", delimiter=',', dtype=None, encoding=None)

# data
lbl = products[1:,0]
num_genes = len(lbl)
id = np.arange(num_genes)
space = products[1:,1].astype(float)
price = products[1:,2].astype(float)
quantity_max = products[1:,3].astype(int)
space_max = 5*10**4

space = space*10**4
space = space.astype(int)
print(space)

rows = np.sum(quantity_max)
cols = space_max
id_repeat = np.repeat(id,quantity_max)
print(quantity_max)
print(id_repeat)

table = np.zeros((rows,cols+1))

for item_index in range(1,rows,1):
    for current_space in range(cols):
        if space[id_repeat[item_index]] > current_space:
            table[item_index,current_space] = table[item_index-1,current_space]
        else:
            table[item_index,current_space] = max([table[item_index-1,current_space],
                                                   table[item_index-1, current_space-space[id_repeat[item_index]]] + price[id_repeat[item_index]]])

pprint(table)