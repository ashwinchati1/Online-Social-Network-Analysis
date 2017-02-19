"""
collect.py
"""

# This file collects data from the twitter using twitter Streaming and REST api...

# import statements...

import tweepy, json, time, sys, os
from tweepy import StreamListener, Stream, OAuthHandler
from TwitterAPI import TwitterAPI
 
# Authentication keys...

consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

authentication = OAuthHandler(consumer_key, consumer_secret)
authentication.set_access_token(access_token, access_token_secret)

# initialize variables

maxTweetsToCollect = 100
tweetsCollectionKeywords = ["Donald Trump","donald trump","realDonaldTrump"]
maxFriendsToCollect = '500'

# Create Authentication...

def auth_user():
    return tweepy.API(authentication)

def get_twitter():
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

#Get tweets using streaming API...

def start_stream():
    collectTweets = Stream(authentication, TweetsCollector())
    collectTweets.filter(track=tweetsCollectionKeywords)

class TweetsCollector(StreamListener):

    def __init__(self, api=None):
        super(TweetsCollector, self).__init__()
        if (os.path.exists('fetched_data/fetched_tweets.txt')):
            os.remove('fetched_data/fetched_tweets.txt')
        self.collectedTweets = 0

    def on_data(self, twitterData):
        if self.collectedTweets < maxTweetsToCollect:
            jsonDataFormat = json.loads(twitterData)
            if jsonDataFormat['text'] is not None:
                if not (jsonDataFormat['text'].startswith('RT') | jsonDataFormat['text'].startswith('rt')):
                    to_append = {
                        "id_str" :  jsonDataFormat['user']['id_str'],
                        "screen_name" : jsonDataFormat['user']['screen_name'],
                        "text" : jsonDataFormat['text'],
                        "location" : jsonDataFormat['user']['location']
                    }
                    with open('fetched_data/fetched_tweets.txt','a') as fetchedTweets:
                        json.dump(to_append,fetchedTweets)
                        fetchedTweets.write('\n')
                    self.collectedTweets = self.collectedTweets + 1
                return(True)
        else:
            return(False)

# Get followers...

def read_screen_names(filename):
    file_open = open(filename,'r')
    screen_name = []
    for name in file_open:
        jsonDataFormat = json.loads(name)
        screenName = jsonDataFormat.get('screen_name')
        if(len(screenName)!=0):
            screen_name.append(screenName.strip())
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

def get_friends(twitter, screen_names):
    if (os.path.exists('fetched_data/user_friends.txt')):
        os.remove('fetched_data/user_friends.txt')

    for name in screen_names:
        friends = robust_request(twitter,'friends/ids',{'screen_name': name, 'count': maxFriendsToCollect})
        friendList = sorted([frnd for frnd in friends])
        with open('fetched_data/user_friends.txt','a') as collectedFreinds:
            entry = {name:friendList}
            json.dump(entry,collectedFreinds)
            collectedFreinds.write('\n')

            
def main():
    api = auth_user()
    start_stream()
    twitter = get_twitter()
    screen_names = read_screen_names('fetched_data/fetched_tweets.txt')
    get_friends(twitter, screen_names)
    
if __name__ == '__main__':
    main()