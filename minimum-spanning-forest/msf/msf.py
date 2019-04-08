################################################################################
#
# File for solving a minimum spanning forest weight approximation
#
# Author: Nik Vaessen
################################################################################

import sys
import os
import random

from typing import Tuple, List
from queue import Queue

# define whether the programming is running in the kattis environment or in
# debug mode
env_name = "RUNNING_IN_DEBUG"
running_in_debug = env_name in os.environ and bool(os.environ[env_name])

if running_in_debug:
    import networkx
    from generate_graphs import get_test_set, weight_key, mst_weight_key

################################################################################
# Implementation of weight approximation algorithm

# define the max possible weight for an edge
max_weight = 4

# parameter for approximating connected components
number_random_vertexes = 20


class GraphOracle:
    """
    An oracle can answer what edges are incident to a particular vertex v
    """

    def __init__(self):
        """
        Initialize by storing an empty dictionary which will store
        all known neighbours
        """
        self.neighbours = {}
        self.n = self._internal_ask_size()

    def ask_neighbour(self, v: int) -> List[Tuple[int, int]]:
        """
        Ask for the neighbours of node v in the graph

        :param v: the node v to request the neighbours of
        :return: a list of a tuple of ints, where the first integer is the neighbour
        vertex and the second int is the cost of the edge between v and the
        neighbour
        """
        if v not in self.neighbours:
            self.neighbours[v] = self._internal_ask_neighbour(v)

        return self.neighbours[v]

    def ask_size(self):
        """
        Ask for the amount of nodes in the graph
        :return: the amount of nodes in the graph
        """
        return self.n

    def _internal_ask_neighbour(self, v):
        """
        Abstract method for asking the neighbouring set of node v
        :return: a list of a tuple of ints, where the first integer is the neighbour
        vertex and the second int is the cost of the edge between v and the
        neighbour
        """
        raise NotImplemented()

    def _internal_ask_size(self):
        """
        Abstract method for asking the size of the graph
        :return: the amount of vertexes in the graph
        """
        raise NotImplemented()


def approx_msf_weight(oracle: GraphOracle):
    """
    Approximate the minimum-spanning forest weight of a graph known by the
    GraphOracle

    :param oracle: the oracle which we can ask questions about the properties of
    the graph
    :return: the approximation of the graph
    """
    size = oracle.ask_size()

    c = 0
    for w in range(1, max_weight):
        c += approx_num_component(oracle, w)

    guess = size - max_weight + c

    if guess > size:
        return guess
    else:
        return size


def approx_num_component(oracle: GraphOracle, w: int):
    """
    Approximate the number of components in a subgraph

    :param oracle: the oracle able to give information about neighbours
    of vertexes
    :param w: only consider neighbours with edges with with w
    :return: the number of components in the subgraph created by only
    considering edges of the given weight
    """
    size = oracle.ask_size()
    vertices = random_sample(0, size, number_random_vertexes)

    b = 0
    for v in vertices:
        max_hops = rv_1_to_inf()

        explore = Queue()
        visited = []
        hops = 0

        for n in get_specific_neighbours(oracle.ask_neighbour(v), w):
            explore.put(n)

        while not explore.empty() and hops < max_hops:
            hops += 1

            visit = explore.get()
            visited.append(visited)

            for n in get_specific_neighbours(oracle.ask_neighbour(visit), w):
                if n not in visited:
                    explore.put(n)

        if explore.empty():
            b += 1

    return (size / len(vertices)) * b

################################################################################
# Utility methods


def get_specific_neighbours(edge_list, w):
    filtered_neighbours = []

    for neighbour, weight in edge_list:
        if w == weight:
            filtered_neighbours.append(neighbour)

    return filtered_neighbours


def random_sample(min: int, max: int, n: int):
    """
    Generate a random sample of n numbers between min and max

    :param min: the minimum number in the sample
    :param max: the maximum number in the sample
    :param n: the amount of numbers in the sample
    :return: a list of n numbers, where each number is between min and max-1
    """
    d = max - min
    sample = [int(min + (d * random.random())) for _ in range(0, n)]

    return sample


def rv_1_to_inf():
    """
    Sample a number x between 1 and inf, where X is a random variable
    with Pr(X=> k] = (1/k)

    :return: a randomly chosen number with the given distribution above
    """
    k = 1
    r = random.random()

    while True:
        p_l = 1 / k
        p_h = 1 / (k + 1)

        if p_l >= r >= p_h:
            return k
        else:
            k += 1

################################################################################
# Logic for solving the problem in kattis


class KattisGraphOracle(GraphOracle):
    """
    Implement the GraphOracle where kattis will give us the answers
    """

    def __init__(self):
        super().__init__()

    def _internal_ask_neighbour(self, v):
        print(v, flush=True)
        line = sys.stdin.readline().split(" ")
        n = int(line[0])
        line = line[1:]

        return [(int(line[2*i]), int(line[2*i + 1])) for i in range(0, n)]

    def _internal_ask_size(self):
        return int(sys.stdin.readline())


def answer_kattis(weight: float):
    """
    Give the answer to the kattis environment

    :param weight: the answer to the weight of the msf problem
    """
    print("end", weight)

################################################################################
# Logic for debugging the code


if running_in_debug:
    class DebugOracle(GraphOracle):
        """
        Pretends to act like an oracle on a graph with a known mst weight
        """

        def __init__(self, graph: networkx.Graph):
            self.G = graph
            super().__init__()

        def _internal_ask_neighbour(self, v):
            n = self.G.neighbors(v)

            answer = []
            for k in n:
                w = self.G[v][k][weight_key]
                answer.append((k, w))

            return answer

        def _internal_ask_size(self):
            return len(self.G)

################################################################################
# Main method if this file is called directly


def main():
    """
    Executes the kattis process of approximation the weight of a minimum
    spanning forest of a graph
    """
    if running_in_debug:
        for G in get_test_set():
            oracle = DebugOracle(G)
            weight = approx_msf_weight(oracle)
            print("real weight: {:5d}".format(G.graph[mst_weight_key]), end=" ")
            print("approx weight: {:5f}".format(weight))
    else:
        oracle = KattisGraphOracle()
        weight = approx_msf_weight(oracle)
        answer_kattis(weight)


if __name__ == '__main__':
    main()
