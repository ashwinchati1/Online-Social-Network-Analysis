"""
classify.py
"""

# This file creates classifier...

# Import statements...

import os, json, string, csv, re
import numpy as np
import urllib.request
import pandas as pd
from scipy.sparse import csr_matrix
from itertools import chain, combinations
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
from collections import Counter, defaultdict
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression

# Initialize global variables...

neg_words = None
pos_words = None
test_data_count = None

# read training data...

def read_training_data():
    link = urllib.request.urlopen('')
    file = ZipFile(BytesIO(link.read()))
    training_data_file = file.open('')

    data = pd.read_csv(training_data_file, header=None, names=['polarity', 'id', 'date','query', 'user', 'text'])

    temp_docs = []
    temp_label = []
    for index, row in data.iterrows():
        if(row['polarity'] == 4 or row['polarity'] == 0):
            if(row['polarity'] == 4):
                temp_label.append(1)
            if(row['polarity'] == 0):
                temp_label.append(0)
            temp_docs.append(row['text'])
    return np.array(temp_docs), np.array(temp_label)

# Read tweets from fetched_tweets.txt file...

def read_test_data():
    tweets_data = []
    if (os.path.exists('fetched_data/fetched_tweets.txt')):
        file_open = open('fetched_data/fetched_tweets.txt','r')
        for name in file_open:
            #print(len(name))
            if(len(name) != 0):
                d = json.loads(name)
                replace_char = re.sub('[^A-Za-z0-9]+', ' ', d['text'])
                tweets_data.append(replace_char)
    global test_data_count
    test_data_count = len(tweets_data)
    return tweets_data

# Download AFINN and get data...

def get_AFINN():
    link_of_AFINN = urlopen('')
    zipfile = ZipFile(BytesIO(link_of_AFINN.read()))
    afinn_file = zipfile.open('')
    words_dict = dict()
    for line in afinn_file:
        parts = line.strip().split()
        if len(parts) == 2:
            words_dict[parts[0].decode("utf-8")] = int(parts[1])
    return words_dict

# Create positive and negative words dictories...

def createPosNegWrdsList(afinn):
    negative_words = []
    positive_words = []

    for key, value in afinn.items():
        if(value < 0):
            negative_words.append(key)
        if(value > 0):
            positive_words.append(key)
    return negative_words, positive_words

# Create tokens from tweets...

def tokenize(doc, keep_internal_punct=False):
    to_remove = string.punctuation
    each_document = doc
    new_array = []

    if(keep_internal_punct==False):
        for char in to_remove:
            each_document = each_document.replace(char," ")
        for str in each_document.lower().split():
            new_array.append(str)
        tokenized_array = np.array(new_array)
    else:
        for str in each_document.lower().split():
            new_array.append(str.lstrip(to_remove).rstrip(to_remove).strip())
        tokenized_array = np.array(new_array)
    return tokenized_array

# token_feature function...

def token_features(tokens, feats):
    count_tokens = Counter(tokens)
    for token in count_tokens:
        feats.update({'token='+token:count_tokens[token]})

def token_pair_features(tokens, feats, k=3):
    token_pair = []
    for i in range(0,tokens.size-k+1):
        pairs = []
        pairs.append(tokens[i])
        for j in range(1,k):
            pairs.append(tokens[i+j])

        for l in range(0,len(pairs)-1):
            for m in range(1,k):
                if(l+m<len(pairs)):
                    token_pair.append(pairs[l]+"__"+pairs[l+m])

    token_pair_frequency = Counter(token_pair)

    for token_pair in token_pair_frequency:
        feats.update({'token_pair='+token_pair:token_pair_frequency[token_pair]})

# create lexicon features...

def lexicon_features(tokens, feats):
    count_negative = 0
    count_positive = 0
    for token in tokens:
        if(token.lower() in neg_words):
            count_negative = count_negative + 1
        if(token.lower() in pos_words):
            count_positive = count_positive + 1
    feats.update({'neg_words':count_negative})
    feats.update({'pos_words':count_positive})

# Features the data...

def featurize(tokens, feature_fns):
    features = defaultdict(lambda: 0)
    for f in feature_fns:
        f(tokens,features)
    return sorted(features.items(), key=lambda x: x[0])

# vectorize the data...

def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    new_vocab = {}
    column_indices = []
    matrix_data = []
    pointer_to_index = [0]
    voc = defaultdict(lambda: 0)
    intermediate_list = list()
    sv = dict()

    for tkn in tokens_list:
        feature_dict = defaultdict(lambda: 0)
        feature_dict = featurize(tkn,feature_fns)
        intermediate_list.append(feature_dict)
        if(vocab==None):
            for entry in feature_dict:
                if(entry[0] in voc): voc.update({entry[0]:voc.get(entry[0])+1})
                else: voc.update({entry[0]:1})

    if(vocab==None):
        for entry in sorted(voc):
            if(voc[entry]>= min_freq):
                new_vocab.update({entry:len(new_vocab)})

    if(vocab==None):
        svi = sorted(new_vocab.items(), key=lambda x: x[0])
    else:
        svi = sorted(vocab.items(), key=lambda x: x[0])

    sv = dict(svi)

    for entry in intermediate_list:
        for element in entry:
            if element[0] in sv:
                column_indices.append(sv.get(element[0]))
                matrix_data.append(element[1])
        pointer_to_index.append(len(column_indices))
    final_matrix = csr_matrix((matrix_data, column_indices, pointer_to_index),dtype='int64', shape = (len(tokens_list),len(sv)))
    return final_matrix, sv

# calculate accuracy score...

def accuracy_score(truth, predicted):
    return len(np.where(truth==predicted)[0]) / len(truth)

# calculate corss_validation_accuracy...

def cross_validation_accuracy(clf, X, labels, k):
    cross_val = KFold(len(labels), k)
    cross_val_accuracies = []
    for training_index, test_index in cross_val:
        clf = LogisticRegression()
        clf.fit(X[training_index], labels[training_index])
        predicted_value = clf.predict(X[test_index])
        acc = accuracy_score(labels[test_index], predicted_value)
        cross_val_accuracies.append(acc)
    average_accuracy = np.mean(cross_val_accuracies)
    return average_accuracy

def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    result_list=[]
    clf = LogisticRegression()

    for punctuation in punct_vals:
        for frequency_value in min_freqs:
            for rng in range(0, len(feature_fns)+1):
                for functions in combinations(feature_fns, rng):
                    if functions:
                        X, voc = vectorize([tokenize(d,punctuation) for d in docs], functions, frequency_value, vocab=None)
                        accuracy = cross_validation_accuracy(clf, X, labels,5)
                        result_list.append({'features':functions,'punct':punctuation,'accuracy':accuracy, 'min_freq':frequency_value})

    return sorted(sorted(result_list,key=lambda x: x['min_freq'], reverse = True),key=lambda x: x['accuracy'], reverse = True)

def fit_best_classifier(docs, labels, best_result):
    clf = LogisticRegression()
    X , vocab = vectorize([tokenize(d, best_result['punct']) for d in docs], best_result['features'], best_result['min_freq'], vocab=None)
    return clf.fit(X,labels), vocab

def parse_test_data(best_result, vocab):

    docs = read_test_data()
    X, vocab = vectorize([tokenize(doc, best_result['punct']) for doc in docs], best_result['features'], best_result['min_freq'], vocab)

    return docs, X

def classified_data(test_docs,X_test, clf, n):
    count_positive = 0
    count_negative = 0
    example_positive = ""
    example_negative = ""
    
    probability_values = clf.predict_proba(X_test)
    predicted_values = clf.predict(X_test)
    predicted_tweets = []
    for prediction in range(len(predicted_values)):
        predicted_tweets.append({'predicted':predicted_values[prediction],'proba':probability_values[prediction],'doc':test_docs[prediction]})
    tweets_sorted = sorted(predicted_tweets,key=lambda x:x['proba'][x['predicted']],reverse=True)
    
    for tweet in tweets_sorted:
        if(tweet['predicted'] == 0):
            count_negative = count_negative + 1
        else:
            count_positive = count_positive + 1
         
    for tweet in tweets_sorted:
        if(tweet['predicted'] == 0 and tweet['proba'][0] > 0.9):
            example_negative = tweet['doc']
            break
    
    for tweet in tweets_sorted:
        if(tweet['predicted'] == 1 and tweet['proba'][1] > 0.9):
            example_positive = tweet['doc']
            break
    return count_positive, count_negative, example_positive, example_negative

def write_to_file(count_positive, count_negative, example_positive, example_negative):
    
    if (os.path.exists('fetched_data/classify_stats.txt')):
        os.remove('fetched_data/classify_stats.txt')

    with open('fetched_data/classify_stats.txt','w') as classify_stats:
        classify_stats.write("Number of instances per class found: \n\tclass negative: %s \n\tclass positive: %s" %(count_negative,count_positive))
        classify_stats.write('\n')
        classify_stats.write("One example from each class: \n\tclass negative: %s \n\tclass positive: %s" %(example_negative,example_positive))
    
def main():
    docs, labels = read_training_data()
    read_test_data()
    posNegWordList = get_AFINN()
    negative_words, positive_words = createPosNegWrdsList(posNegWordList)
    feature_fns = [token_features, token_pair_features, lexicon_features]
    global neg_words
    neg_words = set(negative_words)
    global pos_words
    pos_words = set(negative_words)

    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [3])
    best_result = results[0]
    clf, vocab = fit_best_classifier(docs, labels, results[0])
    test_docs, X_test = parse_test_data(best_result, vocab)
    count_positive, count_negative, example_positive, example_negative = classified_data(test_docs,X_test, clf, 5)
    write_to_file(count_positive, count_negative, example_positive, example_negative)
        
if __name__ == '__main__':
    main()