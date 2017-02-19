"""
cluster.py
"""

import tweepy, json, time, sys, os, pickle
import networkx as nx
import matplotlib.pyplot as plt

#initialize global variabls...

numOfTweets = None
numOfUsers = None

# Read data from user_friends.txt

def read_data():
    friends_list = []
    count_friends = 0
    count_tweets = 0
    if (os.path.exists('fetched_data/user_friends.txt')):
        file_open = open('fetched_data/user_friends.txt','r')
        for name in file_open:
            count_friends = count_friends + 1
            temp_list = []
            d = json.loads(name)
            temp_list.append(list(d.keys())[0])
            temp_list.append(list(d.values())[0])
            friends_list.append(temp_list)

    if (os.path.exists('fetched_data/fetched_tweets.txt')):
        file_open = open('fetched_data/fetched_tweets.txt','r')
        for name in file_open:
            if(len(name) != 0):
                count_tweets = count_tweets + 1
    global numOfTweets, numOfUsers
    numOfTweets =  count_tweets
    numOfUsers = count_friends
    return friends_list

# Create graph containing users and their followers

def create_graph(friends):
    created_graph = nx.Graph()

    for i in range(0, len(friends)):
        created_graph.add_node(friends[i][0])
        
    for user1 in range(0, len(friends)-1):
        for user2 in range(user1+1,len(friends)):
            jaccard = find_jaccard(friends[user1][1],friends[user2][1])
            if(jaccard >= 0.0012):
                created_graph.add_edge(friends[user1][0],friends[user2][0])
    
    grph_wthot_outliers = get_subgraph(created_graph, 1)
    
    nx.draw_networkx(grph_wthot_outliers, node_color='red',edge_color='0.7',node_size=65, with_labels=False)
    plt.axis('off')
    plt.savefig('network.png')
    return grph_wthot_outliers

# Create subgraph...

def get_subgraph(graph, min_degree):
    node_degrees = graph.degree()
    node_list = list()
    count = 0
    for i in graph:
        if graph.degree(i)>=min_degree:
            node_list.append(i)
    sub_graph = graph.subgraph(node_list)
    return sub_graph

# Calculate Jaccard Similarity...

def find_jaccard(user1,user2):
    u1 = set(user1)
    u2 = set(user2)
    try:
        jaccard_similarity = len(u1.intersection(u2)) / len(u1.union(u2))
    except ZeroDivisionError:
        jaccard_similarity = 0
    return jaccard_similarity

clusters = []
def create_clusters(graph):
    if graph.order() in range(3,30):
        clusters.append(graph)
        return
    
    if graph.order() <= 2:
        return
    
    components = [c for c in nx.connected_component_subgraphs(graph)]
    while len(components) == 1:
        remove_edge = get_edge(graph)
        graph.remove_edge(*remove_edge)
        components = [c for c in nx.connected_component_subgraphs(graph)]

    for c in components:
        create_clusters(c)
    return

def get_edge(graph):
    edge = nx.edge_betweenness_centrality(graph)
    sorted_edge = sorted(edge.items(), key=lambda x: x[1], reverse=True)[0][0]
    return sorted_edge


def write_to_file(clusters):

    if (os.path.exists('fetched_data/cluster_stats.txt')):
        os.remove('fetched_data/cluster_stats.txt')

    with open('fetched_data/cluster_stats.txt','w') as collect_stats:
        collect_stats.write("Number of users collected: %s" %numOfUsers)
        collect_stats.write('\n')
        collect_stats.write("Number of messages collected: %s" %numOfTweets)
        collect_stats.write('\n')
        collect_stats.write("Number of communities discovered: %s" %len(clusters))
        collect_stats.write('\n')
        collect_stats.write("Average number of users per community: %s" %round(average_users(clusters)))

def average_users(clusters):
    count_users = 0
    """for clus in range(len(clusters)): 
        for comp in range(len(clusters[clus].nodes())):
            count_users = count_users + 1"""

    for clus in clusters:
        count_users = count_users + len(clus.nodes())
    try:
        count = count_users/len(clusters)
    except ZeroDivisionError:
        count = 0
    return count
    
def main():
    friends = read_data()
    graph = create_graph(friends)
    create_clusters(graph)
    write_to_file(clusters)

if __name__ == '__main__':
    main()