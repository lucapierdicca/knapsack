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


def fitness_single_repair(solution, solution_index):
    return np.sum(solution * price)

def fitness_multi_repair(solution, solution_index):
    w1 = 5/7
    w2 = 1/7
    w3 = 1/7
    f1 = (np.sum(quantity_max * price) - np.sum(solution * price)) / np.sum(quantity_max * price)
    f2 = (np.sum(quantity_max) - np.sum(solution)) / np.sum(quantity_max)
    f3 = (space_max - np.sum(solution * space)) / space_max
    return  -(w1*f1 + w2*f2 + w3*f3)

def fitness_single_death(solution, solution_index):
    if np.sum(solution * space) > space_max:
        return 0.0
    return np.sum(solution * price)

def fitness_single_penalty(solution, solution_index):
    w2 = 6.0
    f1 = np.sum(solution * price) / np.sum(quantity_max * price)
    f2 = np.abs(space_max - np.sum(solution * space)) / (np.sum(quantity_max * space) - space_max)
    return f1 - (w2*f2)

def repair(ga, caller):

    current_pop = None

    if caller == "START":
        current_pop = ga.population
    elif caller == "MUTATION":
        current_pop = ga.last_generation_offspring_mutation

    # for s in current_pop:
    #     if random.random() >= 1.0:
    #         gene_index = 0
    #         while np.sum(s * space) > space_max:
    #             if s[posidx_to_id_asc[gene_index]] > 0 :
    #                 s[posidx_to_id_asc[gene_index]] -= 1
    #             else:
    #                 gene_index+=1
    #     else:
    #         while np.sum(s * space) > space_max:
    #             gene_index = random.randint(0,len(s)-1)
    #             if s[gene_index] > 0 :
    #                 s[gene_index] -= 1

    for s in current_pop:
        while np.sum(s * space) > space_max:
            gene_index = random.randint(0,len(s)-1)
            if s[gene_index] > 0 :
                s[gene_index] -= 1

def on_start(ga):
    repair(ga,"START")

def on_mutation(ga, offspring):
    repair(ga,"MUTATION")

def greedy_solution():
    price_mar = price / space
    ranking = sorted([[pm, i] for pm, i in zip(price_mar, id)], key=lambda x: x[0], reverse=True)

    posidx_to_id_desc = {index: r[1] for index, r in enumerate(ranking)}
    #posidx_to_id_asc = {len(lbl) - 1 - key: value for key, value in posidx_to_id_desc.items()}

    gene_index = 0
    greedy_sol = np.zeros(num_genes, dtype=int)
    while np.sum(greedy_sol * space) <= space_max:
        if greedy_sol[posidx_to_id_desc[gene_index]] < quantity_max[posidx_to_id_desc[gene_index]]:
            greedy_sol[posidx_to_id_desc[gene_index]] += 1
        else:
            gene_index += 1

    greedy_sol[posidx_to_id_desc[gene_index]] -= 1

    return greedy_sol

n_run = 3
spaces_max = [1.0, 2.0]
fitness = [(fitness_single_repair,"fitness_single_repair", on_start, on_mutation),
           (fitness_multi_repair, "fitness_multi_repair", on_start, on_mutation),
           (fitness_single_death, "fitness_single_death", None, None),
           (fitness_single_penalty, "fitness_single_penalty", None, None)]

fitness_data = {}

for f in fitness:
    for i in range(len(spaces_max)):

        data = {"best_solution": [],
                "best_solution_fitness": [],
                "best_solution_price": [],
                "best_solution_space": [],
                "best_solution_generation": []}

        space_max = spaces_max[i]

        print(f[1], space_max)

        # ratio-greedy solution
        greedy_sol = greedy_solution()
        print(greedy_sol, np.sum(greedy_sol*price), np.sum(greedy_sol*space))

        data["space_max"] = space_max
        data["ratio_greedy_solution"] = greedy_sol

        # GA solution
        for i in range(n_run):

            # GA params
            params = {
                "num_generations": 300,
                "num_parents_mating": 4,
                "parent_selection_type": "sss",
                "keep_parents": 1,
                "initial_population": None,
                "sol_per_pop": 10,
                "num_genes": num_genes,
                "gene_type": int,
                "gene_space": [range(q + 1) for q in quantity_max],
                "crossover_type": "two_points",
                "mutation_type": "random",
                "mutation_percent_genes": 10,

                "fitness_func":f[0],
                "on_start":f[2],
                "on_mutation":f[3]}


            # init and start GA
            ga = pygad.GA(**params)
            ga.run()

            # save run data
            best_solution, best_solution_fitness, _ = ga.best_solution()
            data["best_solution"].append(best_solution)
            data["best_solution_fitness"].append(best_solution_fitness)
            data["best_solution_price"].append(np.sum(best_solution*price))
            data["best_solution_space"].append(np.sum(best_solution*space))
            data["best_solution_generation"].append(ga.best_solution_generation)

        if f[1] not in fitness_data:
            fitness_data[f[1]] = [data]
        else:
            fitness_data[f[1]].append(data)



# plot run data

pprint(fitness_data)



