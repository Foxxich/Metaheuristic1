import scipy

from tsplib_parser.tsp_file_parser import TSPParser
from typing import List, Dict
import random
import tsplib95
import pandas as pd
from scipy.spatial import distance
import itertools

from matplotlib import pyplot as plt


def parse_boolean(value: str) -> bool:
    value = value.lower()

    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False

    return False


def euclid_load(file_path: str):
    return TSPParser(filename=file_path, plot_tsp=True).get_cities_dict()


def matrix_load(file_path: str):
    TSPParser(filename=file_path, plot_tsp=True)


def row_load(file_path: str):
    with open(file_path) as f:
        problem = tsplib95.read(f)
    print(problem.display_data)


def euclid_generate(n, width, height):
    cities_dict = {}
    points = []
    for i in range(npoints):
        while True:
            point = (random.randint(0, width), random.randint(0, height))
            if point not in points:
                points.append(point)
                cities_dict[str(i)] = point
                break


def calculate_distance_matrix(dict):
    # for subset in itertools.combinations(cities_dict, 2):
    #     print(subset)
    #     scipy.spatial.distance.cdist(subset[2], subset[3], metric='euclidean')
    #     print(subset[1])
    coord_array = np.asarray(list(dict.values()))
    dist_matrix = distance_matrix(coord_array, coord_array)
    return dist_matrix


def row_generate(n):
    pass


def matrix_generate(n):
    pass


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
            cities_dictionary = euclid_load(file_path)
            calculate_distance_matrix(cities_dictionary)
        elif decision2 == "g":
            npoints = int(input("Type the npoints:"))
            width = float(input("Enter the Width you want:"))
            height = float(input("Enter the Height you want:"))
            euclid_generate(npoints, width, height)
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
