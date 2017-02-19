# coding: utf-8

"""

Collecting a political social network

In this assignment, I've given you a list of Twitter accounts of 4
U.S. presedential candidates.

The goal is to use the Twitter API to construct a social network of these
accounts. We will then use the [networkx](http://networkx.github.io/) library
to plot these links, as well as print some statistics of the resulting graph.

"""

# Imports you'll need.
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
import json
from TwitterAPI import TwitterAPI
from collections import defaultdict

consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

def get_twitter():

    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


def read_screen_names(filename):

    file_open = open(filename,'r')
    screen_name = []
    for name in file_open:
        screen_name.append(name.strip())
    return screen_name


def robust_request(twitter, resource, params, max_tries=5):

    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)


def get_users(twitter, screen_names):

    getUser = robust_request(twitter,' 	/lookup',{'screen_name':screen_names})
    user = [usr for usr in getUser]
    return user


def get_friends(twitter, screen_name):

    friends = robust_request(twitter,'friends/ids',{'screen_name': screen_name, 'count': '5000'})
    friendList = [frnd for frnd in friends]
    return sorted(friendList)


def add_all_friends(twitter, users):

    for usr in users:
        usr['friends'] = get_friends(twitter, usr['screen_name'])


def print_num_friends(users):

    for usr in (sorted(users,key = lambda users:users['screen_name'])):
        print(usr['screen_name'], len(usr['friends']))

def count_friends(users):

    new_dict = []

    for usr in users:
        new_dict.extend(usr['friends'])
    count = Counter(new_dict)
    return count


def friend_overlap(users):

    new_list = []

    for i in range (0,len(users)-1):
        for j in range (i+1,len(users)):
            a = []
            b = []
            a = set(users[i]['friends'])&(set(users[j]['friends']))
            b = (users[i]['screen_name'],users[j]['screen_name'],len(a))
            new_list.append(b)
    return sorted((sorted((sorted(new_list,key = lambda b:b[1])),key = lambda b:b[0])), key = lambda b:b[2],reverse=True)



def followed_by_hillary_and_donald(users, twitter):

    follower_list = []

    for i in range(0,len(users)):
        name = users[i]['screen_name']
        if users[i]['screen_name'] == 'HillaryClinton':
            getfolloweduser = robust_request(twitter,'friends/ids',{'screen_name':name})
            follower_Hillary = [r for r in getfolloweduser]
        if users[i]['screen_name'] == 'realDonaldTrump':
            getfolloweduser = robust_request(twitter,'friends/ids',{'screen_name':name})
            follower_Donald = [r for r in getfolloweduser]

    followers = set(follower_Hillary)&set(follower_Donald)

    followerName = []
    new_list = []
    for i in followers:
        getfolloweduser = robust_request(twitter,'users/lookup',{'user_id':followers})
        followerName = [r for r in getfolloweduser]

    for i in range(0,len(followerName)):
        new_list.append(followerName[i]['screen_name'])
    return new_list

def create_graph(users, friend_counts):

    created_graph = nx.Graph()
    for i in range(0,len(users)):
        created_graph.add_node(users[i]['screen_name'])

    for i in range(0,len(users)):
        for j in (users[i]['friends']):
            if(friend_counts[j]>1):
                created_graph.add_edge(users[i]['screen_name'], j)
    return created_graph


def draw_network(graph, users, filename):

    lable_node = {}

    for i in range(0,len(users)):
        lable_node[users[i]['screen_name']] = users[i]['screen_name']

    draw_graph = nx.draw_networkx(graph, node_color='red',edge_color='0.7',node_size=65,labels=lable_node)
    plt.axis('off')
    plt.savefig(filename)

def main():

    twitter = get_twitter()
    screen_names = read_screen_names('candidates.txt')
    print('Established Twitter connection.')
    print('Read screen names: %s' % screen_names)
    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    print('found %d users with screen_names %s' %
          (len(users), str([u['screen_name'] for u in users])))
    add_all_friends(twitter, users)
    print('Friends per candidate:')
    print_num_friends(users)
    friend_counts = count_friends(users)
    print('Most common friends:\n%s' % str(friend_counts.most_common(5)))
    print('Friend Overlap:\n%s' % str(friend_overlap(users)))
    print('User followed by Hillary and Donald: %s' % followed_by_hillary_and_donald(users, twitter))

    graph = create_graph(users, friend_counts)
    print('graph has %s nodes and %s edges' % (len(graph.nodes()), len(graph.edges())))
    draw_network(graph, users, 'network.png')
    print('network drawn to network.png')


if __name__ == '__main__':
    main()


