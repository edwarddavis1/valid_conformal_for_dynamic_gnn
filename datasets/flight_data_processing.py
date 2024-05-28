# %%
"""
Raw data available: https://zenodo.org/records/5815448#.Y1_ydy-l1hD

As the raw data is large, it has not been included.
This script takes the raw data and saves it as a series of sparse matrices.
"""

import csv
import glob
import numpy as np
from scipy import sparse

# %%
datapath = "Flight Data/"
airport_lats = {}
airport_longs = {}
airport_continents = {}
first = True

# airports.csv from https://ourairports.com/data/

with open(datapath + "airports.csv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=",", quotechar='"')

    for line in reader:
        if first:
            first = False
            continue

        airport = line[1]
        lat = line[4]
        long = line[5]
        continent = line[7]

        if continent not in ["NA", "AS", "EU", "SA", "OC", "AF"]:
            airport_continents[airport] = "XX"
        else:
            airport_continents[airport] = continent

        airport_lats[airport] = float(lat)
        airport_longs[airport] = float(long)


# %%
nodes = {}
count = 0

sources = [[] for i in range(36)]
targets = [[] for i in range(36)]
weights = [[] for i in range(36)]

years = ["2019", "2020", "2021"]
months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

for y in range(len(years)):
    year = years[y]
    for m in range(len(months)):
        month = months[m]

        filename = glob.glob(
            datapath + "flightlist_" + str(year) + str(month) + "*.csv"
        )[0]
        print(filename)
        file = open(filename, "r")

        t = 12 * y + m
        first = True

        for line in file:
            if first:
                first_data = np.array(line.strip("\n").split(","))
                source_idx = np.where(first_data == "origin")[0][0]
                target_idx = np.where(first_data == "destination")[0][0]
                time_idx = np.where(first_data == "firstseen")[0][0]

                first = False
                continue

            data = line.strip("\n").split(",")
            source = data[source_idx]
            target = data[target_idx]

            # Ignore blank entries
            if source == "" or target == "":
                continue

            # Ignore self-connections
            if source == target:
                continue

            # Ignore XX continent flight sources
            if source not in airport_continents or airport_continents[source] == "XX":
                continue

            # Ignore XX continent flight targets
            if target not in airport_continents or airport_continents[target] == "XX":
                continue

            if source not in nodes:
                nodes[source] = count
                count += 1
            if target not in nodes:
                nodes[target] = count
                count += 1

            sources[t].append(nodes[source])
            targets[t].append(nodes[target])

# %%
n = len(nodes)
T = 36
As = []

for t in range(T):
    m = len(sources[t])

    A = sparse.coo_matrix((np.ones(m), (sources[t], targets[t])), shape=(n, n))
    A = A + A.T

    As.append(A)


# %%
Z = []

for node in nodes:
    if node in airport_continents:
        continent = airport_continents[node]
    else:
        continent = "XX"

    Z.append(continent)

Z = np.array(Z)

np.save(datapath + "Z.npy", Z)
np.save(datapath + "nodes.npy", nodes)
