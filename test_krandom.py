from main import random_solution
from main import euclid_load, euclid_generate, euclid_calculate_distance_matrix
from main import matrix_load, matrix_generate
from main import row_load, row_generate
from main import aim_function, load_best_solution, calculate_prd


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

matrix_dict = {"euc2d": [euc_dist_matrix, euc_best_solution], "full_matrix": [full_dist_matrix, full_matrix_best_solution], "row": [row_dist_matrix, row_best_solution]}
for matrix in matrix_dict:
    print(matrix)
    with open(f"JO_krandom_result_100_for_{matrix}.csv", "w") as file:
        for k in range(1, 101):
            print(k, end=";")
            file.write(f"{k};")
            for attempt in range(10):
                result = random_solution(matrix_dict[matrix][0], k)
                print(calculate_prd(result[1], matrix_dict[matrix][1]), end=";")
                file.write(f"{calculate_prd(result[1], matrix_dict[matrix][1])};")
            print()
            file.write("\n")

for matrix in matrix_dict:
    print(matrix)
    with open(f"JO_krandom_result_1000000_for_{matrix}.csv", "w") as file:
        for k in range(100000, 1000001, 50000):
            print(k, end=";")
            file.write(f"{k};")
            for attempt in range(1):
                result = random_solution(matrix_dict[matrix][0], k)
                print(calculate_prd(result[1], matrix_dict[matrix][1]), end=";")
                file.write(f"{calculate_prd(result[1], matrix_dict[matrix][1])};")
            print()
            file.write("\n")

# """ testing for random generate samples """
# print("\n>> Generating Euclides matrix")
# rand_euc_dist_matrix = euclid_generate(n=52, width=1740, height=1175, seed=27)
# rand_euc_estimated_best_solution = 0
# print(rand_euc_dist_matrix)
# print("Best solution: ", rand_euc_estimated_best_solution)
#
#
# print("\n>> Generating Full matrix")
# rand_full_dist_matrix = matrix_generate(n=17, max_cost=74, seed=27)
# rand_full_matrix_estimated_best_solution = 0
# print(rand_full_dist_matrix)
# print("Best solution: ", rand_full_matrix_estimated_best_solution)
#
# print("\n>> Generating DiagRowLow matrix")
# rand_row_dist_matrix = row_generate(n=120, max_cost=1210, seed=27)
# rand_row_estimated_best_solution = 0
# print(rand_row_dist_matrix)
# print("Best solution: ", rand_row_estimated_best_solution)