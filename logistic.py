from numpy import *
import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import random


import tensorflow as tf

pos_review_dir = "review_polarity/txt_sentoken/pos/";
neg_review_dir = "review_polarity/txt_sentoken/neg/";

# -----------------------------------------------------------------------------
# PART 4
# -----------------------------------------------------------------------------
# plot graph part4
def plot_graph_part4(y_test, y_train, y_valid): 
    y_test = np.asarray(y_test)
    y_train = np.asarray(y_train)
    y_valid = np.asarray(y_valid)
    
    plt.subplot(2,1,1)
    plt.plot(y_test, '-', color='red', lw = 2, label="test set")
    plt.plot(y_train, ':', color='green', lw = 2, label=" train set")
    plt.plot(y_valid, '--', color='blue', lw = 2, label=" validation set")
    plt.ylabel('Correctness rate (%)')
    plt.xlabel('Number of iterations')
    plt.title('Learning curves.')
    
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)
           
    plt.show()
    
    
    
def unique_words(pos_indicies, neg_indicies):    
    unique_words_pos = []
    
    # Check every file in positive folder.
    for str in os.listdir(pos_review_dir):
        if ((int(str[2:5])) in pos_indicies):
            for line in open(pos_review_dir + str):
                list_of_words = line.split() # array of words in a single line.
                unique_words_pos = list(set(unique_words_pos + list_of_words));
    
    # print(unique_words_pos);
    print("Number of unique words in positive reviews:", len(unique_words_pos));
        
    unique_words_neg = []
    # Check every file in negative folder.
    for str in os.listdir(neg_review_dir):
        if ((int(str[2:5])) in neg_indicies):
            for line in open(neg_review_dir + str):
                list_of_words = line.split() # array of words in a single line.
                unique_words_neg = list(set(unique_words_neg + list_of_words));
    
    # print(unique_words_neg);
    print("Number of unique words in negative reviews:", len(unique_words_neg));
    
    result = list(set((unique_words_pos + unique_words_neg)))
    return result;
    

def pick_data():
    train_size = 400; 
    test_size =  100; # 100 sample for each of the two classes.
    valid_size = 100;
    
    random.seed(0)
    # positive reviews to be picked.
    pos_to_be_picked = random.sample(range(0, 999), train_size + test_size + valid_size);
    
    random.seed(1)
    # negative reviews to be picked.
    neg_to_be_picked = random.sample(range(0, 999), train_size + test_size + valid_size);
    
    # indices of the positive reviews for the datasets
    train_set_pos = pos_to_be_picked[0:train_size]; 
    valid_set_pos = pos_to_be_picked[train_size: train_size + valid_size];
    test_set_pos = pos_to_be_picked[train_size + valid_size : train_size + valid_size + test_size];
    
    # indices of the positive reviews for the datasets
    train_set_neg = neg_to_be_picked[0:train_size]; 
    valid_set_neg = neg_to_be_picked[train_size: train_size + valid_size];
    test_set_neg = neg_to_be_picked[train_size + valid_size : train_size + valid_size + test_size];
    
    # return list of indices.
    return (train_set_pos, valid_set_pos, test_set_pos, train_set_neg, valid_set_neg, test_set_neg)
    



def get_data(pos_indicies, neg_indicies, unique_words_list):
    n = len(unique_words_list);
    
    get_index = {};
    i = 0;
    for word in unique_words_list:
        get_index[word] = i;
        i = i + 1;
    
    xs = array([])
    ys = array([])
    
    # Buildings the xs and ys for positive reviews
    for str in os.listdir(pos_review_dir):
        if ((int(str[2:5])) in pos_indicies):
            x = zeros(n)
            
            list_of_words = []; # List of words in a single file
            for line in open(pos_review_dir + str):
                list_of_words += line.split() # array of words in a single line
            
            # if appear in the document, change the corresnponding attributes to 1.
            for word in set(list_of_words):
                if not (word in unique_words_list):
                    continue
                idx = get_index[word]
                x[idx] = 1
                
            
            if xs.size == 0:
                xs = vstack([x]);
            elif xs.size > 0:
                xs = vstack([xs, x]);
                
            one_hot = zeros(2)
            one_hot[0] = 1 # positive review ==> y[0] = 1
            
            if ys.size == 0:
                ys = vstack([one_hot])
            elif ys.size != 0:
                ys = vstack([ys, one_hot])
                
                
    # Buildings the xs and ys for negative reviews
    for str in os.listdir(neg_review_dir):
        if ((int(str[2:5])) in neg_indicies):
            x = zeros(n)
            
            list_of_words = []; # List of words in a single file
            for line in open(neg_review_dir + str):
                list_of_words += line.split() # array of words in a single line
            
            # if appear in the document, change the corresnponding attributes to 1.
            for word in set(list_of_words):
                if not (word in unique_words_list):
                    continue
                idx = get_index[word]
                x[idx] = 1
                
            
            if xs.size == 0:
                xs = vstack([x]);
            elif xs.size > 0:
                xs = vstack([xs, x]);
                
            one_hot = zeros(2)
            one_hot[1] = 1 # negative review ==> y[1] = 1
            
            if ys.size == 0:
                ys = vstack([one_hot])
            elif ys.size != 0:
                ys = vstack([ys, one_hot])
                
    return xs,ys
    
    

def part4():
    ll = pick_data();
    
    train_pos = ll[0]; 
    train_neg = ll[3];
    valid_pos = ll[1];
    valid_neg = ll[4];
    test_pos = ll[2];
    test_neg = ll[5];
    
    # Get list of the unique words in the training set.
    list_of_unique_words_train = unique_words(train_pos, train_neg)
    
    # Number of features in a sample
    number_of_features = len(list_of_unique_words_train)
    
    print("Number of unique words in the training set: {}".format(number_of_features));
    
    get_index = {}
    i = 0;
    for word in list_of_unique_words_train:
        get_index[word] = i;
        i = i + 1;
    
    train_x, train_y = get_data(train_pos, train_neg, list_of_unique_words_train);
    valid_x, valid_y = get_data(valid_pos, valid_neg, list_of_unique_words_train);
    test_x, test_y = get_data(test_pos, test_neg, list_of_unique_words_train);
    
    
    # ------------------- TENSOR FLOW ----------------------------
    x = tf.placeholder(tf.float32, [None, number_of_features])
    
    
    W0 = tf.Variable(tf.random_normal([number_of_features, 2], stddev=0.01))
    b0 = tf.Variable(tf.random_normal([2], stddev=0.01))
    
    layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)

    y = tf.nn.softmax(layer1) # computed result
    y_ = tf.placeholder(tf.float32, [None, 2]) # expect result
    
    lam = 1000 # 5 so 0
    decay_penalty =lam*tf.reduce_sum(tf.square(W0)) #+lam*tf.reduce_sum(tf.square(W1))
    reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty
    
    train_step = tf.train.AdamOptimizer(0.00050).minimize(reg_NLL) # training.
    
    init = tf.global_variables_initializer()
    
    sess = tf.Session()
    sess.run(init)
    
    # Gets the correct answer.
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    
    # array for plotting graph
    y_test = [] 
    y_train = []
    y_valid = []
    
    for i in range(200):
        sess.run(train_step, feed_dict={x: train_x, y_: train_y})

        train_set_performance = sess.run(accuracy, feed_dict={x: train_x, y_: train_y});
        valid_set_performance = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y});
        test_set_performance = sess.run(accuracy, feed_dict={x: test_x, y_: test_y});
        
        print("i=", i)
        print("Performance on the test set: ", test_set_performance)
        print("Performance on the train set: ", train_set_performance)
        print("Performance on the validation set:",  valid_set_performance)
        
        y_test.append(test_set_performance * 100)
        y_train.append(train_set_performance * 100)
        y_valid.append(valid_set_performance * 100)
        
    
    plot_graph_part4(y_test, y_train, y_valid);
    
#part4()
        