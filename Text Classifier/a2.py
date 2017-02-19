# coding: utf-8

"""

In this assignment, you will build a text classifier to determine whether a
movie review is expressing positive or negative sentiment. The data come from
the website IMDB.com.


"""

# No imports allowed besides these.
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request


def download_data():
    """ Download and unzip data.
    DONE ALREADY.
    """
    url = ''
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


def read_data(path):

    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


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

neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])

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

def featurize(tokens, feature_fns):

    features = defaultdict(lambda: 0)

    for f in feature_fns:
        f(tokens,features)

    return sorted(features.items(), key=lambda x: x[0])


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
    final_matrix = csr_matrix((matrix_data, column_indices, pointer_to_index),dtype='int64')
    return final_matrix, sv

def accuracy_score(truth, predicted):

    return len(np.where(truth==predicted)[0]) / len(truth)


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

def plot_sorted_accuracies(results):

    sorted_results = results.copy()
    sorted_accuracies = []
    
    sorted_list = sorted(sorted_results,key=lambda x: x['accuracy'])
    for result in sorted_list:
        sorted_accuracies.append(result['accuracy'])
    plt.plot(sorted_accuracies)
    plt.xlabel('setting')
    plt.ylabel('accuracy')
    plt.savefig('accuracies.png')

def mean_accuracy_per_setting(results):

    intermediate_list = {}
    count_list = {}
    final_list = {}
    
    for feature in results:
        ftr = str(feature['features'])
        if("token_features" in ftr and "token_pair_features" not in ftr and "lexicon_features" not in ftr): feat_ftr = "features=token_features"
            
        if("token_features" not in ftr and "token_pair_features" in ftr and "lexicon_features" not in ftr): feat_ftr = "features=token_pair_features"
            
        if("token_features" not in ftr and "token_pair_features" not in ftr and "lexicon_features" in ftr): feat_ftr = "features=lexicon_features"
            
        if("token_features" in ftr and "token_pair_features" in ftr and "lexicon_features" not in ftr ): feat_ftr = "features=token_features token_pair_features"
            
        if("token_features" in ftr and "token_pair_features" not in ftr and "lexicon_features" in ftr): feat_ftr = "features=token_features lexicon_features"
            
        if("token_features" not in ftr and "token_pair_features" in ftr and "lexicon_features" in ftr): feat_ftr = "features=token_pair_features lexicon_features"
            
        if("token_features" in ftr and "token_pair_features" in ftr and "lexicon_features" in ftr): feat_ftr = "features=token_features token_pair_features lexicon_features"
        
        if(feat_ftr not in intermediate_list):
            intermediate_list.update({feat_ftr:feature['accuracy']})
            count_list.update({feat_ftr:1})
        else:
            intermediate_list.update({feat_ftr: intermediate_list.get(feat_ftr)+feature['accuracy']})
            count_list.update({feat_ftr:count_list.get(feat_ftr)+1})
        
        punc_ftr = "punct="+str(feature['punct'])
        if(punc_ftr not in intermediate_list):
            intermediate_list.update({punc_ftr:feature['accuracy']})
            count_list.update({punc_ftr:1})
        else:
            intermediate_list.update({punc_ftr: intermediate_list.get(punc_ftr)+feature['accuracy']})
            count_list.update({punc_ftr:count_list.get(punc_ftr)+1})
       
        freq_ftr = "min_freq="+str(feature['min_freq'])
        if(freq_ftr not in intermediate_list):
            intermediate_list.update({freq_ftr:feature['accuracy']})
            count_list.update({freq_ftr:1})
        else:
            intermediate_list.update({freq_ftr: intermediate_list.get(freq_ftr)+feature['accuracy']})
            count_list.update({freq_ftr:count_list.get(freq_ftr)+1})

    for i in intermediate_list:
        final_list.update({intermediate_list.get(i)/count_list.get(i):i})
    
    return (sorted(final_list.items(),key=lambda x: x[0], reverse = True))
        
def fit_best_classifier(docs, labels, best_result):

    clf = LogisticRegression()
    X , vocab = vectorize([tokenize(d, best_result['punct']) for d in docs], best_result['features'], best_result['min_freq'], vocab=None)
    return clf.fit(X,labels), vocab


def top_coefs(clf, label, n, vocab):

    sotred_vocab = sorted(vocab.items(), key=lambda x: x[1])
    sorted_vocab_list = list(key for key,value in sotred_vocab)
    vocab_array = np.array(sorted_vocab_list)
    
    coefficient = clf.coef_[0]

    if(label == 0): top_indexes = np.argsort(coefficient)[:n]

    if(label == 1): top_indexes = np.argsort(coefficient)[::-1][:n]

    features = vocab_array[top_indexes]
    top_coefficients = abs(coefficient[top_indexes])

    return (sorted([entry for entry in zip(features, top_coefficients)],key=lambda x:x[1],reverse=True))



def parse_test_data(best_result, vocab):

    docs, labels = read_data(os.path.join('data', 'test'))
    X, vocab = vectorize([tokenize(doc, best_result['punct']) for doc in docs], best_result['features'], best_result['min_freq'], vocab)

    return docs, labels, X


def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    
    incorrect_values = list()
    probability_values = clf.predict_proba(X_test)
    predicted_values = clf.predict(X_test)
    
    for prediction in range(len(predicted_values)):
        if(predicted_values[prediction] != test_labels[prediction]):
            intermediate_list = dict()
            intermediate_list.update({'truth':test_labels[prediction],'predicted':predicted_values[prediction],'proba':probability_values[prediction],'doc':test_docs[prediction]})
            incorrect_values.append(intermediate_list)
    incorrect_list_sorted = sorted(incorrect_values,key=lambda x:x['proba'][x['predicted']],reverse=True)
    for doc in range(0,n):
        print("\ntruth=%d predicted=%d proba=%f \n%s" % (incorrect_list_sorted[doc]['truth'],incorrect_list_sorted[doc]['predicted'],incorrect_list_sorted[doc]['proba'][incorrect_list_sorted[doc]['predicted']],incorrect_list_sorted[doc]['doc']))
    

def main():

    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))

    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
         accuracy_score(test_labels, predictions))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)




if __name__ == '__main__':
    main()
