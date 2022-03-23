from py2opt.routefinder import RouteFinder

from tsplib_parser.tsp_file_parser import TSPParser
from typing import List, Dict
import random
import tsplib95
import pandas as pd
from scipy.spatial import distance_matrix
import itertools
from random import randint
import numpy as np
from array import *
import re


def euclid_load(file_path: str):
    cities_dict = TSPParser(filename=file_path, plot_tsp=True).get_cities_dict()
    return euclid_calculate_distance_matrix(cities_dict)


def euclid_generate(n, width, height):
    cities_dict = {}
    points = []
    for i in range(n):
        while True:
            point = (random.randint(0, width), random.randint(0, height))
            if point not in points:
                points.append(point)
                cities_dict[str(i)] = point
                break
    return euclid_calculate_distance_matrix(cities_dict)
    # todo: round distance matrix to int


def euclid_calculate_distance_matrix(dict):
    coord_array = np.asarray(list(dict.values()))
    return distance_matrix(coord_array, coord_array)


def matrix_load(file_path: str):
    with open(file_path) as f:
        problem = tsplib95.read(f)
    n = problem.dimension
    data_list = []
    f = open(file_path)  # is it needed? in 33 line we open
    points = f.readlines()[7: -1]
    for i in range(0, 2 * n, 2):
        line = points[i].strip() + "," + points[i + 1].strip()
        row = [line]
        for j in range(len(row)):
            for r in (("    ", ","), ("   ", ","), (" ", ",")):
                row[j] = row[j].replace(*r)
            row_int = map(int, row[j].split(','))
            data_list.append(list(row_int))
    print(data_list)
    return data_list


def matrix_generate(n, max_cost):
    data_list = []
    row = []
    for i in range(n):
        for j in range(n):
            if i == j:
                row.append(9999)
            else:
                row.append(randint(1, max_cost))
        data_list.append(row)
        row = []
    return data_list


def row_load(file_path: str):
    with open(file_path) as f:
        text = f.read()
        n = int(text.split("DIMENSION: ")[1].split("\n")[0])
        costs = text.split("EDGE_WEIGHT_SECTION")[1].split("DISPLAY_DATA_SECTION")[0].replace("\n", "").split(" ")[1:]

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


def row_generate(n, max_cost):
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
    solutions = []
    costs = []
    for i in range(k):
        while True:
            solution = tuple(np.random.permutation(cities))
            if solution not in solutions:
                solutions.append(solution)
                costs.append(aim_function(list(solution), dist_matrix))
                break
    index = min(range(len(costs)), key=costs.__getitem__)
    print(index)
    print(solutions[index])
    print(costs[index])


# neighbour
def neighbour_solution(dist_matrix, start):
    """Nearest neighbor algorithm.
    A is an NxN array indicating distance between N locations
    start is the index of the starting location
    Returns the path and cost of the found solution
    """
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
    print(path)
    print(cost)


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
    print(index)
    print(paths[index])
    print(costs[index])


# 2-opt
def two_opt_solution(dist_mat):
    cities_names = []
    for i in range(len(dist_mat)):
        cities_names.append(i)
    route_finder = RouteFinder(dist_mat, cities_names, iterations=len(dist_mat) + 1)  # todo: check number of iterations
    best_distance, best_route = route_finder.solve()

    print(best_distance)
    print(best_route)


if __name__ == '__main__':
    print("Choose data type:")
    print("[1]. euc_2d\n" +
          "[2]. lower_diag_row\n" +
          "[3]. full_matrix")
    decision1 = input()

    print("Press [l] to load data from file OR \n" +
          "Press [g] to generate random instance")
    decision2 = input()

    dist_matrix = None
    if decision1 == "1":
        if decision2 == "l":
            print("Type file path:")
            file_path = input()
            dist_matrix = euclid_load(file_path)
        elif decision2 == "g":
            npoints = int(input("Type the npoints: "))
            width = float(input("Enter the Width you want: "))
            height = float(input("Enter the Height you want: "))
            dist_matrix = euclid_generate(npoints, width, height)
        # print("Points: ", cities_dictionary)
    elif decision1 == "2":
        if decision2 == "l":
            print("Type file path:")
            file_path = input()
            dist_matrix = row_load(file_path)
        elif decision2 == "g":
            npoints = int(input("Type the npoints: "))
            max_cost = int(input("Type max cost: "))
            dist_matrix = row_generate(npoints, max_cost)
    elif decision1 == "3":
        if decision2 == "l":
            print("Type file path:")
            file_path = input()
            dist_matrix = matrix_load(file_path)
        elif decision2 == "g":
            npoints = int(input("Type the npoints: "))
            max_cost = int(input("Type max cost: "))
            dist_matrix = matrix_generate(npoints, max_cost)
    else:
        print("Error")

    print("\nDistance matrix:\n", dist_matrix)
    permutation = [int(x) for x in input("Type permutation split by space: ").split()]
    cost = aim_function(permutation, dist_matrix)
    print(cost)
    print("Choose algorithm:")
    print("[1]. k-random\n" +
          "[2]. neighbour\n" +
          "[3]. neighbour modified\n" +
          "[4]. 2-opt")
    chosen_algorithm = input()
    if chosen_algorithm == "1":
        k = int(input("Type k:"))
        random_solution(dist_matrix, k)
    elif chosen_algorithm == "2":
        v = int(input("Type vertex:"))
        neighbour_solution(dist_matrix, v)
    elif chosen_algorithm == "3":
        neighbour_modified_solution(dist_matrix)
    elif chosen_algorithm == "4":
        two_opt_solution(dist_matrix)
    else:
        print("Error")
