################################################################################
# This file can be used to generate a random graph, calculate the MST weight
# and store all the information in a file
#
# Author: Nik Vaessen
################################################################################

import networkx
import random
import os

from typing import List

################################################################################
# Methods to generate a random graph and calculate the weight of the mst

# key attribute for saving the weight of an edge
weight_key = "weight"

# key attribute for saving the weight of the minimum-spanning tree of a graph
mst_weight_key = "mst_key"


def generate_weighted_graph(n: int,
                            p_edge: float,
                            min_w: int,
                            max_w: int,
                            natural_weights=False) -> networkx.Graph:
    """
    Generate a random weighted graph with only a limited amount of
    possibilities for weights

    :param n: the amount of vertexes in the graph
    :param p_edge: the probability of two vertexes sharing an edge
    :param min_w: the minimum weight which can be assigned to an edge
    :param max_w the maximum weight which can be assigned to an edge
    :param natural_weights whether the weight can be only be a natural number or
    any real value

    :return: a randomly generated graph with the given properties
    """
    G = networkx.generators.gnp_random_graph(n, p_edge)

    for u, v, a in G.edges(data=True):
        if natural_weights:
            w = random.randint(min_w, max_w)
        else:
            w = min_w + (max_w - min_w) * random.random()

        a[weight_key] = w

    return G


def calculate_mst_weight(graph: networkx.Graph):
    """
    Calculate the weight of the minimum spanning tree of a graph

    :param graph: the graph
    :return: the minimum spanning tree of the graph
    """
    T = networkx.algorithms.minimum_spanning_tree(graph, "weight")

    weight = 0
    for u, v, a in T.edges(data=True):
        weight += a[weight_key]

    return weight

################################################################################
# Methods to generate a test set, which will be the default functionality of the
# file


rootdir = os.path.join("../")
sample_dir = os.path.join(rootdir, "samples")


def maybe_create_test_set():
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    if len(os.listdir(sample_dir)) >= 16:
        return

    for n in [10, 100, 500, 1000, 5000]:
        print("####", n, "####")
        for p in [0.1, 0.2, 0.5, 0.8]:
            print("####", p, "####")
            G = generate_weighted_graph(n, p, 1, 4, natural_weights=True)
            weight = calculate_mst_weight(G)
            print(weight)
            G.graph[mst_weight_key] = weight

            fn = "../samples/n{}_p{}_natural_graph.gz".format(n, p)
            networkx.write_gpickle(G, fn)


def get_test_set() -> List[networkx.Graph]:
    """
    Load the generated test set
    :return: a list of graphs
    """
    graphs = []
    for f in os.listdir(sample_dir):
        f = os.path.join(sample_dir, f)
        print("loading ", f)
        graphs.append(networkx.read_gpickle(f))
    print()

    return graphs


def main():
    maybe_create_test_set()

    graphs = get_test_set()
    for g in graphs:
        print(len(g), end=" --> ")
        print(g.graph[mst_weight_key])

        if len(g) == 10:
            for v in range(0, len(g)):
                n = g.neighbors(v)
                print(v, end=" neighbours to ")
                for k in n:
                    print(k, " with weight ", g[v][k][weight_key], end=", ")
                print()


if __name__ == '__main__':
    main()
