# coding: utf-8

# community detection and link prediction algorithms using Facebook "like" data.

# You should not use any imports not listed here:
from collections import Counter, defaultdict, deque
import copy
import math
import networkx as nx
import urllib.request


## Community Detection

def example_graph():

    g = nx.Graph()
    g.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('D', 'E'), ('D', 'F'), ('D', 'G'), ('E', 'F'), ('G', 'F')])
    return g

def bfs(graph, root, max_depth):

    input_queue = deque()
    input_queue.append(root)
    visited_nodes = set()
    node2distances = dict()
    node2num_paths = dict()
    node2parents = dict()
    node2distances.update({root:0})
    node2num_paths.update({root:1})
    visited_nodes.update(root)

    while input_queue:
        node_to_pop = input_queue.popleft()
        node_neighbors = sorted(graph.neighbors(node_to_pop), key = lambda x:x, reverse = True)
        for node in node_neighbors:
            if node not in visited_nodes:
                distance = node2distances.get(node_to_pop)+1
                if distance <= max_depth:
                    visited_nodes.add(node)
                    node2distances.update({node:distance})
                    node2num_paths.update({node:node2num_paths.get(node_to_pop)})
                    parents = []
                    parents.append(node_to_pop)
                    n = graph.neighbors(node)
                    for j in n:
                        if j in visited_nodes and j != node_to_pop and node2distances.get(j)==node2distances.get(node_to_pop):
                            parents.append(j)
                            node2num_paths.update({node:node2num_paths.get(node) + 1})
                    node2parents.update({node:parents})
                    input_queue.append(node)

    return(node2distances,node2num_paths,node2parents)


def complexity_of_bfs(V, E, K):

    return V+E


def bottom_up(root, node2distances, node2num_paths, node2parents):

    sorted_distance = sorted(sorted(node2distances.items(),key=lambda x: x[0], reverse=False),key=lambda x: x[1], reverse=True)
    node_weights = dict()
    edge_weights = dict()

    for node in sorted_distance:
        node_weights.update({node[0]:1})

    for i in sorted_distance:
        parents = node2parents.get(i[0])
        for j in parents or []:
            if len(parents) > 1:
                weight_to_add = node_weights.get(i[0])/len(parents)
                node_weights.update({j:float(node_weights.get(j)+weight_to_add)})
                if i[0] < j: edge_weights.update({(i[0],j):float(weight_to_add)})
                else: edge_weights.update({(j,i[0]):float(weight_to_add)})
            else:
                node_weights.update({j:float(node_weights.get(j)+node_weights.get(i[0]))})
                if i[0] < j: edge_weights.update({(i[0],j):float(node_weights.get(i[0]))})
                else: edge_weights.update({(j,i[0]):float(node_weights.get(i[0]))})

    return edge_weights


def approximate_betweenness(graph, max_depth):

    all_nodes = graph.nodes()
    approx_betweenness = dict()

    for node in all_nodes:
        distance, path, parent = bfs(graph,node,max_depth)
        result_bottom_up = bottom_up(node, distance, path, parent)
        for j in result_bottom_up:
            if not approx_betweenness.get(j): approx_betweenness.update({j:result_bottom_up.get(j)})
            else: approx_betweenness.update({j:approx_betweenness.get(j)+result_bottom_up.get(j)})

    for i in approx_betweenness:
        approx_betweenness.update({i:approx_betweenness.get(i)/2})
    return approx_betweenness

def is_approximation_always_right():

    return "no"


def partition_girvan_newman(graph, max_depth):

    input_graph = graph.copy()
    betweenness_values = sorted(approximate_betweenness(graph,max_depth).items(), key=lambda x: (-x[1], x[0]))
    components_result = [cmp for cmp in nx.connected_component_subgraphs(input_graph)]
    while(len(components_result)==1):
        edge_to_remove = betweenness_values.pop(0)
        input_graph.remove_edge(edge_to_remove[0][0],edge_to_remove[0][1])
        components_result = [cmp for cmp in nx.connected_component_subgraphs(input_graph)]
    return components_result

def get_subgraph(graph, min_degree):

    node_degrees = graph.degree()
    node_list = list()
    count = 0

    for i in graph:
        if graph.degree(i)>=min_degree:
            node_list.append(i)

    sub_graph = graph.subgraph(node_list)
    return sub_graph


def volume(nodes, graph):

    edges = graph.edges(nodes)
    count_of_edges = 0

    for edge in edges:
        count_of_edges = count_of_edges + 1

    return count_of_edges


def cut(S, T, graph):

    count_of_edges = 0

    for i in S:
        for j in T:
            if graph.has_edge(i,j):
                count_of_edges = count_of_edges + 1

    return count_of_edges

def norm_cut(S, T, graph):

    cut_value = cut(S, T, graph)
    vol_of_s = volume(S, graph)
    vol_of_T = volume(T, graph)

    normalized_cut = (cut_value/vol_of_s) + (cut_value/vol_of_T)

    return normalized_cut

def score_max_depths(graph, max_depths):

    result_list =list()
    for depth in max_depths:
        comp = partition_girvan_newman(graph,depth)
        result_value = norm_cut(comp[0].nodes(),comp[1].nodes(),graph)
        result_list.append((depth,result_value))
    return result_list

## Link prediction

# we will remove 5 of the accounts that Bill Gates likes and
# compute our accuracy at recovering those links.

def make_training_graph(graph, test_node, n):

    input_graph = graph.copy()
    to_remove = sorted(input_graph.neighbors(test_node))

    for num_edge in range(0,n):
        input_graph.remove_edge(test_node,to_remove[num_edge])

    return input_graph

def jaccard(graph, node, k):

    neighbor1 = set(graph.neighbors(node))
    jaccard_coefficient = []
    nodelist = []

    for i in graph.nodes():
        neighbor2 = set(graph.neighbors(i))

        if (i not in neighbor1):
            calculate = 1. * len(neighbor1 & neighbor2) / len(neighbor1 | neighbor2)
            if calculate != 1.0:
                jaccard_coefficient.append(((node,i),calculate))

    jaccard_coefficient = sorted(sorted(jaccard_coefficient,key=lambda x: x[0], reverse=False),key=lambda x: x[1], reverse=True)

    for i in range(0,k):
         nodelist.append(jaccard_coefficient[i])

    return nodelist


# One limitation of Jaccard is that it only has non-zero values for nodes two hops away.
#
# Implement a new link prediction function that computes the similarity between two nodes $x$ and $y$  as follows:
#
# $$
# s(x,y) = \beta^i n_{x,y,i}
# $$
#
# where
# - $\beta \in [0,1]$ is a user-provided parameter
# - $i$ is the length of the shortest path from $x$ to $y$
# - $n_{x,y,i}$ is the number of shortest paths between $x$ and $y$ with length $i$


def path_score(graph, root, k, beta):

    nodes = graph.nodes()
    result = list()
    final_result = list()
    distance, paths, parents = bfs(graph,root,float("inf"))

    for node in nodes:
        if node != root and not graph.has_edge(root,node):
            similarity = float(math.pow(beta,distance.get(node))*paths.get(node))
            if similarity != 1.0: result.append(((root,node),similarity))

    return sorted(sorted(result, key=lambda x: x[0], reverse=False),key=lambda x: x[1], reverse=True)[:k]



def evaluate(predicted_edges, graph):

    count_of_edges = 0

    for edge in predicted_edges:
        if graph.has_edge(edge[0],edge[1]): count_of_edges = count_of_edges + 1

    return count_of_edges/len(predicted_edges)


"""
Next, we'll download a real dataset to see how our algorithm performs.
"""
def download_data():

    urllib.request.urlretrieve('', 'edges.txt.gz')


def read_graph():

    return nx.read_edgelist('edges.txt.gz', delimiter='\t')


def main():

    download_data()
    graph = read_graph()
    print('graph has %d nodes and %d edges' %
          (graph.order(), graph.number_of_edges()))
    subgraph = get_subgraph(graph, 2)
    print('subgraph has %d nodes and %d edges' %
          (subgraph.order(), subgraph.number_of_edges()))


    print('norm_cut scores by max_depth:')
    print(score_max_depths(subgraph, range(1, 5)))

    clusters = partition_girvan_newman(subgraph, 3)
    print('first partition: cluster 1 has %d nodes and cluster 2 has %d nodes' %
          (clusters[0].order(), clusters[1].order()))
    print('cluster 2 nodes:')
    print(clusters[1].nodes())


    test_node = 'Bill Gates'
    train_graph = make_training_graph(subgraph, test_node, 5)
    print('train_graph has %d nodes and %d edges' %
          (train_graph.order(), train_graph.number_of_edges()))

    jaccard_scores = jaccard(train_graph, test_node, 5)
    print('\ntop jaccard scores for Bill Gates:')
    print(jaccard_scores)
    print('jaccard accuracy=%g' %
          evaluate([x[0] for x in jaccard_scores], subgraph))

    path_scores = path_score(train_graph, test_node, k=5, beta=.1)
    print('\ntop path scores for Bill Gates for beta=.1:')
    print(path_scores)
    print('path accuracy for beta .1=%g' %
          evaluate([x[0] for x in path_scores], subgraph))


if __name__ == '__main__':
    main()
