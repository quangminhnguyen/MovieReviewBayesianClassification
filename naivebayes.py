from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import os 
import random
import operator
# -----------------------------------------------------------------------------
# PART 0 picking the data
# -----------------------------------------------------------------------------
def pick_data():
    train_size = 400; # 400 samples for each of the two classes.
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
    
#pick_data()

pos_review_dir = "review_polarity/txt_sentoken/pos/";
neg_review_dir = "review_polarity/txt_sentoken/neg/";

# -----------------------------------------------------------------------------
# PART 1
# -----------------------------------------------------------------------------
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


# pos_indicies: list of indicies of positive review.
# neg_indicies: list of indicies of negative review.
# word: word to get the count for.
# return (number of occurences in positive review, number of occurences in negative reviews.) 
def count_word(pos_indicies, neg_indicies, word):

    
    count_pos = 0;
    # Check every file in positive folder.
    for str in os.listdir(pos_review_dir):
        if ((int(str[2:5])) in pos_indicies):
            list_of_words = []
            for line in open(pos_review_dir + str):
                list_of_words += line.split() # array of words in a single line.
            if (word in list_of_words):
                count_pos += 1;
                    
    
    
    count_neg = 0;
    # Check every file in negative folder.
    for str in os.listdir(neg_review_dir):
        if ((int(str[2:5])) in neg_indicies):
            list_of_words = [];
            for line in open(neg_review_dir + str):
                list_of_words += line.split() # array of words in a single line.
            if word in list_of_words:
                count_neg += 1;
                    
                    
    return (count_pos, count_neg)

def part1():
    ll = pick_data();
    
    train_pos = ll[0];
    train_neg = ll[3];
    valid_pos = ll[1];
    valid_neg = ll[4];
    test_pos = ll[2];
    test_neg = ll[5];
    
   
    print("-------Train set-----------");
    list_of_unique_words_train = unique_words(train_pos, train_neg);
    print("Total number of unique words", len(list_of_unique_words_train));
    
    print("------Validation Set------------");
    list_of_unique_words_valid = unique_words(valid_pos, valid_neg);
    print("Total number of unique words", len(list_of_unique_words_valid));
    
    print("------Test Set------------");
    list_of_unique_words_test = unique_words(test_pos, test_neg);
    print("Total number of unique words", len(list_of_unique_words_test));
    
    
    bad = "bad";
    boring = "boring";
    enjoyed = "enjoyed";
    print("----------bad-----------")
    count = count_word(train_pos, train_neg, bad);
    count_bad_pos = count[0]
    count_bad_neg = count[1]
    print("positive reviews count: {}, negative review count: {}".format(count_bad_pos, count_bad_neg));
    
    
    print("----------boring-----------")
    count = count_word(train_pos, train_neg, boring);
    count_boring_pos = count[0]
    count_boring_neg = count[1]
    print("positive reviews count: {}, negative review count: {}".format(count_boring_pos, count_boring_neg));
    
    
    print("----------enjoyed-----------")
    count = count_word(train_pos, train_neg, enjoyed);
    count_enjoyed_pos = count[0]
    count_enjoyed_neg = count[1]
    print("positive reviews count: {}, negative review count: {}".format(count_enjoyed_pos, count_enjoyed_neg));
#part1();



# -----------------------------------------------------------------------------
# PART 2
# -----------------------------------------------------------------------------
# pos_indicies : indicies of positive reviews in original dataset to be test.
# neg_indicies : indicies of negative reviews in original dataset to be test.
# p_pos: list of SCALED P(word in doc | positive review) for every word. [Dict]
# p_neg: list of SCALED P(word in doc | negative review) for every word. [Dict]
# p_pos_opp: list of P(word not in doc | positive review) for every word [Dict]
# p_neg_opp: list of P(word not in doc | negative review) for every word [Dict]
# size: number of samples for each class.
def test_performance(pos_indicies, neg_indicies, p_pos, p_neg, p_pos_opp, p_neg_opp,size):
    
    count_pos_result = 0;
    count_neg_result = 0;
    for str in os.listdir(pos_review_dir):
        if ((int(str[2:5])) in pos_indicies):
            
            for line in open(pos_review_dir + str):
                list_of_words = []; # List of words in a file
                for line in open(pos_review_dir + str):
                    list_of_words += line.split() # array of words in a single line
                
            log_sum_pos = 0;
            log_sum_neg = 0;
            
            for word in list_of_words:
                if word in p_pos: # ignore words that are not in the training set.
                    log_sum_pos += log(p_pos[word])
                if word in p_neg: # ignore words that are not in the training set.
                    log_sum_neg += log(p_neg[word])
            
            
            # Adding log(P(word not in document | Class)) doesn't help to improve the performance
            # so I choose not to include them. You may comment out 5 lines above, and uncomment 
            # 7 lines below to try it out.
            
            # for word in p_pos:
            #    if word in list_of_words:
            #        log_sum_pos += log(p_pos[word])
            #        log_sum_neg += log(p_neg[word])
            #    elif not (word in list_of_words):
            #        log_sum_pos += log(p_pos_opp[word]) 
            #        log_sum_neg += log(p_neg_opp[word]) 

            result_pos = exp(log_sum_pos + log(0.5)); 
            result_neg = exp(log_sum_neg + log(0.5));

            if (result_pos > result_neg):
                count_pos_result += 1;
            
    performance_positive =  ((float(count_pos_result)/size) * 100)
    
    
    for str in os.listdir(neg_review_dir):
        if ((int(str[2:5])) in neg_indicies):
            
            for line in open(neg_review_dir + str):
                list_of_words = []; # List of words in a file
                for line in open(neg_review_dir + str):
                    list_of_words += line.split() # array of words in a single line
                
            log_sum_pos = 0;
            log_sum_neg = 0;
            
            for word in list_of_words:
                if word in p_pos: # ignore words that are not in training set.
                    log_sum_pos += log(p_pos[word])
                if word in p_neg: # ignore words that are not in training set.
                    log_sum_neg += log(p_neg[word])
            
            # Adding log(P(word not in document | Class)) doesn't help to improve the performance
            # so I choose not to include them. You may comment out 5 lines above, and uncomment 
            # 7 lines below to try it out.
            
            # for word in p_neg: 
            #     if word in list_of_words:
            #         log_sum_pos += log(p_pos[word])
            #         log_sum_neg += log(p_neg[word])
            #     elif not (word in list_of_words):
            #         log_sum_pos += log(p_pos_opp[word]) 
            #         log_sum_neg += log(p_neg_opp[word]) 
            
            result_pos = exp(log_sum_pos + log(0.5)); 
            result_neg = exp(log_sum_neg + log(0.5));
            
            
            if (result_pos < result_neg):
                count_neg_result += 1;
                
    performance_negative = ((float(count_neg_result)/size) * 100)
    
    overall_result = (performance_positive + performance_negative) / 2;
    print("Performance on positie reviews: {}".format(performance_positive))
    print("Performance on negative reviews: {}".format(performance_negative))
    print("Overall result: {}").format(overall_result)
            
def part2():
    ll = pick_data();
    
    train_pos = ll[0];
    train_neg = ll[3];
    valid_pos = ll[1];
    valid_neg = ll[4];
    test_pos = ll[2];
    test_neg = ll[5];

    k = 3;
    m = 460.0;
        
    # Get list of the unique words in the training set.
    list_of_unique_words_train = unique_words(train_pos, train_neg);
    
    count_pos = {}
    count_neg = {}
    for word in list_of_unique_words_train:
        count_pos[word] = 0
        count_neg[word] = 0
    
    

    # Check every file in positive folder.
    for str in os.listdir(pos_review_dir):
        if ((int(str[2:5])) in train_pos):
            
            list_of_words = []; # List of words in a file
            for line in open(pos_review_dir + str):
                list_of_words += line.split() # array of words in a single line
                
            for word in set(list_of_words): # looping through the list of words in the file.
                if not (word in count_pos):
                    print "error";
                count_pos[word] += 1; # count(word, +)
    
    
    # Check every file in positive folder.
    for str in os.listdir(neg_review_dir):
        if ((int(str[2:5])) in train_neg):
            
            list_of_words = []; # List of words in a file
            for line in open(neg_review_dir + str):
                list_of_words += line.split() # array of words in a single line
            
            for word in set(list_of_words): # looping through the list of words in the file.
                if not (word in count_neg):
                    print "error";
                count_neg[word] += 1;  # count(word, -);
   
    #print count_neg
    #print count_pos
    
    
    p_pos = {}
    p_neg = {}
    
    p_pos_opp = {}
    p_neg_opp = {}
    for word in list_of_unique_words_train:
        # P(word not in document | positive review)
        p_pos_opp[word] = 1 - (count_pos[word]/400)
        
        # P(word not in document | negative review)
        p_neg_opp[word] = 1 - (count_neg[word]/400) 
        
        p_pos[word] = (count_pos[word] + m * k)/ (400 * k); # 400 = no. of positive reviews in training set.
        p_neg[word] = (count_neg[word] + m * k)/ (400 * k); # 400 = no. of negatives reviews in training set.
         
    
    print("-------------------Validation set performance-----------------")
    test_performance(valid_pos, valid_neg, p_pos, p_neg, p_pos_opp, p_neg_opp,100); # 100 samples for each positive and negative reviews.
    
    print("-------------------Test set performance-----------------")
    test_performance(test_pos, test_neg, p_pos, p_neg, p_pos_opp, p_neg_opp, 100); # 100 samples for each positive and negative reviews.
    
    print("-------------------Train set performance-----------------")
    test_performance(train_pos, train_neg, p_pos, p_neg, p_pos_opp, p_neg_opp, 400); # 100 samples for each positive and negative reviews.   
#part2()





# -----------------------------------------------------------------------------
# PART 3
# -----------------------------------------------------------------------------
def part3():
    ll = pick_data();
    
    train_pos = ll[0];
    train_neg = ll[3];
    valid_pos = ll[1];
    valid_neg = ll[4];
    test_pos = ll[2];
    test_neg = ll[5];

    k = 3;
    m = 460.0;
        
    # Get list of the unique words in the training set.
    list_of_unique_words_train = unique_words(train_pos, train_neg);
    
    count_pos = {}
    count_neg = {}
    for word in list_of_unique_words_train:
        count_pos[word] = 0
        count_neg[word] = 0
    
    

    # Check every file in positive folder.
    for str in os.listdir(pos_review_dir):
        if ((int(str[2:5])) in train_pos):
            
            list_of_words = []; # List of words in a file
            for line in open(pos_review_dir + str):
                list_of_words += line.split() # array of words in a single line
                
            for word in set(list_of_words): # looping through the list of words in the file.
                if not (word in count_pos):
                    print "error";
                count_pos[word] += 1; # count(word, +)
    
    
    # Check every file in positive folder.
    for str in os.listdir(neg_review_dir):
        if ((int(str[2:5])) in train_neg):
            
            list_of_words = []; # List of words in a file
            for line in open(neg_review_dir + str):
                list_of_words += line.split() # array of words in a single line
            
            for word in set(list_of_words): # looping through the list of words in the file.
                if not (word in count_neg):
                    print "error";
                count_neg[word] += 1;  # count(word, -);
   
    #print count_neg
    #print count_pos
    
    
    p_pos = {}
    p_neg = {}
    for word in list_of_unique_words_train:
        p_pos[word] = (count_pos[word] + m * k)/ (400 * k); # 400 = no. of positive reviews in training set.
        p_neg[word] = (count_neg[word] + m * k)/ (400 * k); # 400 = no. of negatives reviews in training set.
    
    
    diff = {}
    for word in p_pos:
        diff[word] = p_pos[word] - p_neg[word];
    
    print "Top 10 keywords for positive review."
    top10_pos = dict(sorted(diff.iteritems(), key=operator.itemgetter(1), reverse=True)[:10])
    
    print "Top 10 keywords for negative review."
    top10_neg = dict(sorted(diff.iteritems(), key=operator.itemgetter(1), reverse=False)[:10])
    
    print top10_pos
    print top10_neg

#part3();