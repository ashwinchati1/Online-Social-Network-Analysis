# coding: utf-8

# Recommendation systems
#

from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():

    url = ''
    urllib.request.urlretrieve(url, '')
    zfile = zipfile.ZipFile('')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):

    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):

    data_frame = movies['genres']
    genere_list = []
    
    for items in data_frame:
        genere_list.append(tokenize_string(items))

    to_append = pd.DataFrame({'tokens': genere_list})
    movies = movies.join(to_append)
    
    return movies

def featurize(movies):

    token_detail = []
    intermediate_vocab = {}
    final_vocab = {}
    vocab_list = []
    csr_list = []
    tf = []
    df = {}

    for row in movies['tokens']:
        token_detail.append(row)

    for entry in token_detail:
        l = {}
        for token in entry:
            l.update({token:entry.count(token)})
        tf.append(l)
    
    for entry in token_detail:
        for genere in entry:
            if(genere in intermediate_vocab):
                df.update({genere:df.get(genere)+1})
            else:
                intermediate_vocab.update({genere:0})
                vocab_list.append(genere)
                df.update({genere:1})
    
    vocab_list = sorted(vocab_list)

    index = 0
    for vocab in vocab_list:
        final_vocab.update({vocab:index})
        index = index + 1

    ind = 0
    for entry in token_detail:
        matrix_data = []
        column_indices = []
        pointer_to_index = [0]
        
        for genere in entry:
            if genere in final_vocab:
                tfidf = (tf[ind].get(genere) / tf[0].get(max(tf[0]))) * math.log10(len(token_detail)/df.get(genere))
                matrix_data.append(tfidf)
                column_indices.append(final_vocab.get(genere))
        ind = ind + 1
        pointer_to_index.append(len(column_indices))
        final_matrix = csr_matrix((matrix_data, column_indices, pointer_to_index),shape=(1,len(final_vocab)))
        csr_list.append(final_matrix)

    to_append = pd.DataFrame({'features': csr_list})
    movies = movies.join(to_append)
    
    return (movies,final_vocab)
    
def train_test_split(ratings):

    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):

    dot_product = a.dot(b.transpose())
    
    copy_of_a = a.copy()
    copy_of_b = b.copy()
    copy_of_a.data **=2
    copy_of_b.data **=2
    
    cosine_result = dot_product / (np.sqrt(copy_of_a.sum()) * np.sqrt(copy_of_b.sum()))
    cosine_array = cosine_result.toarray()
    return float(cosine_array[0])
    
def make_predictions(movies, ratings_train, ratings_test):

    to_return = []
    for index, row_test in ratings_test.iterrows():
        sum_cosine = 0
        sum_rating = 0
        count_rating = 0
        sum_crm = 0
        feature_test = movies[movies['movieId'].isin([row_test['movieId']])]['features'].iloc[0]
        for index, row_train in ratings_train[ratings_train.userId == row_test.userId ].iterrows():
            feature_train = movies[movies['movieId'].isin([row_train['movieId']])]['features'].iloc[0]
            cosine_result = cosine_sim(feature_test,feature_train)
            rt = row_train['rating']
            sum_rating = sum_rating + rt
            count_rating = count_rating + 1
            if(cosine_result > 0):
                sum_cosine = sum_cosine + cosine_result
                sum_crm = sum_crm + (cosine_result * rt)
        if(sum_cosine == 0):
            to_return.append(sum_rating/count_rating)
        else:
            to_return.append(sum_crm/sum_cosine)
    na = np.array(to_return)
    return na
            
            
def mean_absolute_error(predictions, ratings_test):

    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])

if __name__ == '__main__':
    main()
