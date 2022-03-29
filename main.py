import random
import time
from math import factorial
from random import randint

import numpy as np
import tsplib95
from py2opt.routefinder import RouteFinder
from scipy.spatial import distance_matrix

from tsplib_parser.tsp_file_parser import TSPParser


# todo: add seed to random

def euclid_load(file_path: str):
    cities_dict = TSPParser(filename=file_path, plot_tsp=True).get_cities_dict()
    return euclid_calculate_distance_matrix(cities_dict)


def euclid_generate(n, width, height, seed):
    random.seed(seed)
    cities_dict = {}
    points = []
    if n > width * height:
        print(f"ERROR: Maximum number of points is {width * height} ({n} given), setting n = {width * height}")
        n = width * height
    for i in range(n):
        while True:
            point = (random.randint(0, width), random.randint(0, height))
            if point not in points:
                points.append(point)
                cities_dict[str(i)] = point
                break
    return euclid_calculate_distance_matrix(cities_dict)


def euclid_calculate_distance_matrix(dict):
    coord_array = np.asarray(list(dict.values()))
    x = distance_matrix(coord_array, coord_array)
    return np.rint(x)


def matrix_load(file_path: str):
    with open(file_path) as f:
        problem = tsplib95.read(f)
    f = open(file_path)
    points = f.readlines()[7: -1]
    n = problem.dimension
    data_list = []
    for i in range(0, 2 * n, 2):
        line = points[i].strip() + "," + points[i + 1].strip()
        row = [line]
        for j in range(len(row)):
            for r in (("    ", ","), ("   ", ","), (" ", ",")):
                row[j] = row[j].replace(*r)
            row_int = map(int, row[j].split(','))
            data_list.append(list(row_int))
    return data_list


def matrix_generate(n, max_cost, seed):
    random.seed(seed)
    data_list = []
    row = []
    for i in range(n):
        for j in range(n):
            if i == j:
                row.append(0)
            else:
                row.append(randint(1, max_cost))
        data_list.append(row)
        row = []
    return data_list


def row_load(file_path: str):
    with open(file_path) as f:
        text = f.read()
        n = int(text.split("DIMENSION: ")[1].split("\n")[0])
        costs = text.split("EDGE_WEIGHT_SECTION")[1].split("DISPLAY_DATA_SECTION")[0].replace("\n", ' ').replace("  ",
                                                                                                                 " ").split(
            " ")[1:]

    data_list = np.zeros((n, n))
    index = 0
    for i in range(n):
        for j in range(i):
            data_list[j][i] = int(costs[index])
            data_list[i][j] = data_list[j][i]
            index += 1
        data_list[i][i] = 0
        index += 1
    return data_list


def row_generate(n, max_cost, seed):
    random.seed(seed)
    data_list = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            data_list[i][j] = randint(1, max_cost)
            data_list[j][i] = data_list[i][j]
        data_list[i][i] = 0
    return data_list


def aim_function(permutation, dist_matrix):
    cost = 0
    for i in range(len(permutation)):
        if i == (len(permutation) - 1):
            cost += dist_matrix[permutation[i]][permutation[0]]
        else:
            cost += dist_matrix[permutation[i]][permutation[i + 1]]
    return cost


# algorithms

def random_solution(dist_matrix, k):
    cities = list(range(len(dist_matrix)))
    solutions = set()
    best_solution = set()
    best_solution_cost = -1
    if k > factorial(len(dist_matrix)):
        print(f"WARNING: there is {factorial(len(dist_matrix))} different solutions (you want to test {k}), setting k = {factorial(len(dist_matrix))}")
        k = factorial(len(dist_matrix))
    for i in range(k):
        while True:
            solution = tuple(np.random.permutation(cities))
            if solution not in solutions:
                # if i != 0 and i % 1000 == 0:
                #     print(f"Searched {i} solutions...")
                solutions.add(solution)
                cost = aim_function(list(solution), dist_matrix)
                if best_solution_cost == -1 or best_solution_cost > cost:
                    best_solution = solution
                    best_solution_cost = cost
                break
            else:
                print("Solution repeat")
    # print(f"Route: {best_solution}")
    # print(f"Cost: {best_solution_cost}")
    return [best_solution, best_solution_cost]


# neighbour
def neighbour_solution(dist_matrix, start):
    """Nearest neighbor algorithm.
    A is an NxN array indicating distance between N locations
    start is the index of the starting location
    Returns the path and cost of the found solution
    """
    # print({len(dist_matrix)},"  ", {start})
    path = [start]
    cost = 0
    dist_matrix = np.array(dist_matrix)
    N = dist_matrix.shape[0]
    mask = np.ones(N, dtype=bool)  # boolean values indicating which
    # locations have not been visited
    mask[start] = False

    for i in range(N - 1):
        last = path[-1]
        next_ind = np.argmin(dist_matrix[last][mask])  # find minimum of remaining locations
        next_loc = np.arange(N)[mask][next_ind]  # convert to original location
        path.append(next_loc)
        mask[next_loc] = False
        cost += dist_matrix[last, next_loc]
    return cost


# neighbour modified
def neighbour_modified_solution(dist_matrix):
    paths = []
    costs = []
    """Nearest neighbor algorithm.
    dist_matrix is an NxN array indicating distance between N locations
    start is the index of the starting location
    Returns the path and cost of the found solution
    """
    for start in range(len(dist_matrix)):
        path = [start]
        cost = 0
        dist_matrix = np.array(dist_matrix)
        N = dist_matrix.shape[0]
        mask = np.ones(N, dtype=bool)  # boolean values indicating which
        # locations have not been visited
        mask[start] = False

        for i in range(N - 1):
            last = path[-1]
            next_ind = np.argmin(dist_matrix[last][mask])  # find minimum of remaining locations
            next_loc = np.arange(N)[mask][next_ind]  # convert to original location
            path.append(next_loc)
            mask[next_loc] = False
            cost += dist_matrix[last, next_loc]
        paths.append(path)
        costs.append(cost)
    index = min(range(len(costs)), key=costs.__getitem__)
    # print(f"Best solution found in {index} neighbour")
    print(f"Route: {paths[index]}")
    print(f"Cost: {costs[index]}")


# 2-opt
def two_opt_solution(dist_mat):
    cities_names = []
    for i in range(len(dist_mat)):
        cities_names.append(i)
    route_finder = RouteFinder(dist_mat, cities_names, iterations=len(dist_mat) + 1)
    best_distance, best_route = route_finder.solve()

    print(f"Route: {best_route}")
    print(f"Cost: {best_distance}")


def load_best_solution(data_to_search):  # todo - add for calculate_prd
    f = open("tsp_best_solutions.txt")
    lines = f.readlines()
    data_to_search = data_to_search.replace('.atsp', '')
    data_to_search = data_to_search.replace('.tsp', '')
    solution = ''
    for i in range(len(lines)):
        if data_to_search in lines[i]:
            solution = lines[i]
    solution = solution.split(":")[1]
    return int(solution)


def calculate_prd(result, opt_result):
    return (result - opt_result) / opt_result * 100


if __name__ == '__main__':
    while (True):
        print("Choose data type:")
        print("[1]. euc_2d\n" +
              "[2]. lower_diag_row\n" +
              "[3]. full_matrix")
        decision1 = input()

        test = False
        best_solution = 0

        print("Press [l] to load data from file OR \n" +
              "Press [g] to generate random instance\n" +
              "Press [t] to test")
        decision2 = input()

        dist_matrix = None
        if decision1 == "1":
            if decision2 == "l":
                print("Type file path:")
                file_path = input()
                dist_matrix = euclid_load(file_path)
                best_solution = load_best_solution(file_path)

            elif decision2 == "g":
                npoints = int(input("Type the npoints: "))
                width = int(input("Enter the Width you want: "))
                height = int(input("Enter the Height you want: "))
                seed = int(input("Enter seed for random generator: "))
                dist_matrix = euclid_generate(npoints, width, height, seed)

            elif decision2 == "t":
                test = True
                npoints = int(input("Type the npoints: "))
                width = float(input("Enter the Width you want: "))
                height = float(input("Enter the Height you want: "))
                seed = int(input("Enter seed for random generator: "))
                for i in range(int(npoints / 2), npoints):
                    dist_matrix = euclid_generate(i, i, i, seed)
            # print("Points: ", cities_dictionary)
        elif decision1 == "2":
            if decision2 == "l":
                print("Type file path:")
                file_path = input()
                dist_matrix = row_load(file_path)
                best_solution = load_best_solution(file_path)
            elif decision2 == "g":
                npoints = int(input("Type number of cities: "))
                max_cost = int(input("Type max cost: "))
                seed = int(input("Enter seed for random generator: "))
                dist_matrix = row_generate(npoints, max_cost, seed)
            elif decision2 == "t":
                test = True
                npoints = int(input("Type the npoints: "))
                width = float(input("Enter the Width you want: "))
                height = float(input("Enter the Height you want: "))
                seed = int(input("Enter seed for random generator: "))
                for i in range(int(npoints / 2), npoints):
                    dist_matrix = euclid_generate(i, i, i, seed)
        elif decision1 == "3":
            if decision2 == "l":
                print("Type file path:")
                file_path = input()
                dist_matrix = matrix_load(file_path)
                best_solution = load_best_solution(file_path)
            elif decision2 == "g":
                npoints = int(input("Type the npoints: "))
                max_cost = int(input("Type max cost: "))
                seed = int(input("Enter seed for random generator: "))
                dist_matrix = matrix_generate(npoints, max_cost, seed)
            elif decision2 == "t":
                test = True
                npoints = int(input("Type the npoints: "))
                width = float(input("Enter the Width you want: "))
                height = float(input("Enter the Height you want: "))
                seed = int(input("Enter seed for random generator: "))
                for i in range(int(npoints / 2), npoints):
                    dist_matrix = euclid_generate(i, i, i, seed)
        else:
            print("Error")

        if not test:
            print("\nDistance matrix:\n", dist_matrix)
            permutation = [int(x) for x in input("Type permutation split by space: ").split()]
            cost = aim_function(permutation, dist_matrix)
            print(f"Cost: {cost}")
            prd = calculate_prd(cost, best_solution)
            print(f"PRD: {prd}")

        go_back = False
        while not go_back:
            print("\nChoose algorithm:")
            print("[1]. k-random\n" +
                  "[2]. neighbour\n" +
                  "[3]. neighbour modified\n" +
                  "[4]. 2-opt\n" +
                  "[5]. get another TSP instance\n" +
                  "[6]. neighbour with different k\n" +
                  "[7]  all\n")
            chosen_algorithm = input()
            if chosen_algorithm == "1":
                k = int(input("Type k:"))
                start = time.time()
                random_solution(dist_matrix, k)
                end = time.time()
                print("Time elapsed: ", end - start)
            elif chosen_algorithm == "2":
                v = int(input("Type vertex:"))
                start = time.time()
                neighbour_solution(dist_matrix, v)
                end = time.time()
                print("Time elapsed: ", end - start)
            elif chosen_algorithm == "3":
                start = time.time()
                neighbour_modified_solution(dist_matrix)
                end = time.time()
                print("Time elapsed: ", end - start)
            elif chosen_algorithm == "4":
                start = time.time()
                two_opt_solution(dist_matrix)
                end = time.time()
                print("Time elapsed: ", end - start)
            elif chosen_algorithm == "5":
                go_back = True
                print("\n\n")
            elif chosen_algorithm == "6":
                for i in range(len(dist_matrix)):
                    start = time.time()
                    neighbour_solution(dist_matrix, i)
                    end = time.time()
                    print("Time elapsed: ", end - start)
                    print("----------------------------")
            elif chosen_algorithm == "7":
                k = int(input("Type k:"))

                start = time.time()
                random_solution(dist_matrix, k)
                end = time.time()
                print("Time elapsed for k: ", end - start)

                v = int(input("Type vertex:"))
                start = time.time()
                neighbour_solution(dist_matrix, v)
                end = time.time()
                print("Time elapsed for neighbour: ", end - start)

                start = time.time()
                neighbour_modified_solution(dist_matrix)
                end = time.time()
                print("Time elapsed for neighbour modified: ", end - start)

                start = time.time()
                two_opt_solution(dist_matrix)
                end = time.time()
                print("Time elapsed for 2opt: ", end - start)
            else:
                print("Error")