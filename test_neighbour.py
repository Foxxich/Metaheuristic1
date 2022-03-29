import csv

import numpy as np

from main import random_solution, neighbour_solution
from main import euclid_load, euclid_generate, euclid_calculate_distance_matrix
from main import matrix_load, matrix_generate
from main import row_load, row_generate
from main import aim_function, load_best_solution, calculate_prd
from random import choice

""" testing for tsplib samples """
print(">> Loading Euclides matrix")
euc_dist_matrix = euclid_load("berlin52.tsp")
euc_best_solution = load_best_solution("berlin52.tsp")
print(euc_dist_matrix)
print("Best solution: ", euc_best_solution)

print("\n>> Loading Full matrix")
full_dist_matrix = matrix_load("br17.atsp")
full_matrix_best_solution = load_best_solution("br17.atsp")
print(full_dist_matrix)
print("Best solution: ", full_matrix_best_solution)

print("\n>> Loading DiagRowLow matrix")
row_dist_matrix = row_load("gr120.tsp")
row_best_solution = load_best_solution("gr120.tsp")
print(row_dist_matrix)
print("Best solution: ", row_best_solution)

matrix_dict = {"euc2d": [euc_dist_matrix, euc_best_solution],
               "full_matrix": [full_dist_matrix, full_matrix_best_solution],
               "row": [row_dist_matrix, row_best_solution]}
new_lis = list(matrix_dict.items())
con_arr = np.array(new_lis)
j = 0
for matrix in matrix_dict:
    cond = con_arr[j][1][0]
    j += 1
    with open(f"VL_neighbour_result_for_{matrix}.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["N", "Result"])
        for i in range(len(cond)):
            # print(matrix)
            # print(len(matrix_dict[matrix][0]))
            print(i, end=";")
            for attempt in range(1):
                result = neighbour_solution(matrix_dict[matrix][0], i)
                print(calculate_prd(result, matrix_dict[matrix][1]), end=";")
                writer = csv.writer(file)
                writer.writerow([i, calculate_prd(result, matrix_dict[matrix][1])])
            print()
