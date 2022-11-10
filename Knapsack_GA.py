import numpy as np
import pygad
from pprint import pprint
import random

# read data
products = np.genfromtxt("products.csv", delimiter=',', dtype=None, encoding=None)

# problem data
lbl = products[1:,0]
num_genes = len(lbl)
id = np.arange(num_genes)
space = products[1:,1].astype(float)
price = products[1:,2].astype(float)
quantity_max = products[1:,3].astype(int)
space_max = 1


def fitness_single_repair(solution, solution_index):
    return np.sum(solution * price)

def fitness_multi_repair(solution, solution_index):
    w1 = 1.0
    w2 = 1.0
    w3 = 1.0
    f1 = np.sum(solution * price) / np.sum(quantity_max * price)
    f2 = np.sum(solution) / np.sum(quantity_max)
    f3 = np.sum(solution * space) / space_max

    return w1*f1 + w2*f2 + w3*f3

def fitness_single_death(solution, solution_index):
    if np.sum(solution * space) > space_max:
        return 0.0
    return np.sum(solution * price)

def fitness_single_penalty(solution, solution_index):
    f1 = 1.0 / (np.abs(80000 - np.sum(solution*price)) + 0.00001)
    f3 = 1.0 / (np.abs(space_max - np.sum(solution*space)) + 0.00001)
    return 100*f1 + f3




def repair(ga, caller):

    current_pop = None

    if caller == "START":
        current_pop = ga.population
    elif caller == "MUTATION":
        current_pop = ga.last_generation_offspring_mutation

    for s in current_pop:
        if random.random() >= 1.0:
            k = 0
            while np.sum(s * space) > space_max:
                if s[id_to_pos_asc[k]] > 0 :
                    s[id_to_pos_asc[k]] -= 1
                else:
                    k+=1
        else:
            while np.sum(s * space) > space_max:
                k = random.randint(0,len(s)-1)
                if s[k] > 0 :
                    s[k] -= 1

def on_start(ga):
    repair(ga,"START")

def on_mutation(ga, offspring):
    repair(ga,"MUTATION")

# ratio-greedy solution
price_mar = price/space
ranking = sorted([[pm,qm,s,p,i] for pm,qm,s,p,i in zip(price_mar,quantity_max,space,price,id)],
                 key=lambda x:x[0], reverse=True)

id_to_pos_desc = {index:r[4] for index,r in enumerate(ranking)}
id_to_pos_asc = {len(lbl)-1-key:value for key,value in id_to_pos_desc.items()}

greedy_q_max = np.array([r[1] for r in ranking])
greedy_space = np.array([r[2] for r in ranking])
greedy_price = np.array([r[3] for r in ranking])
greedy_sol = np.zeros(num_genes, dtype=int)

gene_index = 0
while np.sum(greedy_sol*greedy_space) < space_max:
    if greedy_sol[gene_index] < greedy_q_max[gene_index]:
        greedy_sol[gene_index]+=1
    else:
        gene_index+=1

greedy_sol[gene_index]-=1

init_sol = np.zeros(num_genes, dtype=int)
for i in range(num_genes):
    init_sol[ranking[i][4]] = greedy_sol[i]

print(init_sol, np.sum(init_sol*price), np.sum(init_sol*space))


params = {

    "num_generations": 300,

    "num_parents_mating": 4,
    "parent_selection_type": "sss",
    "keep_parents": 1,

    "initial_population": None,
    "sol_per_pop": 20,

    "num_genes": num_genes,
    "gene_type": int,
    "gene_space": [range(q + 1) for q in quantity_max],

    "crossover_type": "two_points",

    "mutation_type": "random",
    "mutation_percent_genes": 10,

    "save_best_solutions" : False,
    "save_solutions":False,

    "fitness_func":fitness_multi_repair,
    "on_start":on_start,
    "on_mutation":on_mutation}


ga = pygad.GA(**params)
ga.run()

best_sol, best_price, gen = ga.best_solution()
print(best_sol, np.sum(best_sol*price), np.sum(best_sol*space), ga.best_solution_generation)
for s,f in zip(ga.best_solutions, ga.best_solutions_fitness):
    print(s,f)


# ga.plot_fitness()


