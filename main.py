from tsplib_parser.tsp_file_parser import TSPParser
from typing import List, Dict
import random
import tsplib95
import pandas as pd
from scipy.spatial import distance_matrix
import itertools
import numpy as np

def euclid_load(file_path: str):
    cities_dict = TSPParser(filename=file_path, plot_tsp=True).get_cities_dict()
    return calculate_distance_matrix(cities_dict)


def matrix_load(file_path: str):
    with open(file_path) as f:
        problem = tsplib95.read(f)
    n = problem.dimension
    data_list = []
    f = open(file_path)
    points = f.readlines()[7: -1]
    j = 0
    for i in range(0, 2 * n, 2):
        data_list.append(points[i].strip() + " " + points[i + 1].strip())
        print(data_list[j])
        j += 1


def row_load(file_path: str):
    with open(file_path) as f:
        problem = tsplib95.read(f)
    print(problem.display_data)


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
    return calculate_distance_matrix(cities_dict)


def calculate_distance_matrix(dict):
    coord_array = np.asarray(list(dict.values()))
    dist_matrix = distance_matrix(coord_array, coord_array)
    return dist_matrix


def row_generate(n, width, height):
    pass


def matrix_generate(n, width, height):
    pass

def aim_function(permutation, dist_matrix):
    cost = 0
    for i in range(len(permutation)):
        if i == (len(permutation) - 1):
            print(i, 0)
            cost += dist_matrix[permutation[i]][permutation[0]]
            print(cost)
        else:
            print(i, i+1)
            cost += dist_matrix[permutation[i]][permutation[i+1]]
            print(cost)
    return cost



if __name__ == '__main__':
    print("Choose data type:")
    print("[1]. euc_2d\n" +
          "[2]. lower_diag_row\n" +
          "[3]. full_matrix")
    decision1 = input()

    print("Press [l] to load data from file OR \n" +
          "Press [g] to generate random instance")

    decision2 = input()

    if decision1 == "1":
        if decision2 == "l":
            print("Type file path")
            file_path = input()
            dist_matrix = euclid_load(file_path)
        elif decision2 == "g":
            npoints = int(input("Type the npoints:"))
            width = float(input("Enter the Width you want:"))
            height = float(input("Enter the Height you want:"))
            dist_matrix = euclid_generate(npoints, width, height)
        print("\nDistance matrix:\n", dist_matrix)
        # print("Points: ", cities_dictionary)
        permutation = [int(x) for x in input("Type permutation: ").split()]
        cost = aim_function(permutation, dist_matrix)
        print(cost)
    elif decision1 == "2":
        if decision2 == "l":
            print("Type file path")
            file_path = input()
            row_load(file_path)
        elif decision2 == "g":
            npoints = int(input("Type the npoints:"))
            width = float(input("Enter the Width you want:"))
            height = float(input("Enter the Height you want:"))
            row_generate(npoints, width, height)
    elif decision1 == "3":
        if decision2 == "l":
            print("Type file path")
            file_path = input()
            matrix_load(file_path)
        elif decision2 == "g":
            npoints = int(input("Type the npoints:"))
            width = float(input("Enter the Width you want:"))
            height = float(input("Enter the Height you want:"))
            matrix_generate(npoints, width, height)
    else:
        print("Error")