# -----------------------------------------------------------------------------
# PART 6
# -----------------------------------------------------------------------------

from logistic import *
from naivebayes import *

def count_A_in_B (A, B):
    count1 = 0;
    for word in A:
        if word in B:
            count1 = count1 + 1;
    
    count2 = 0;
    for word in B:
        if word in A:
            count2 = count2 + 1;
    
    return count1, count2

def words_in_both(A, B):
    rs = []
    for word1 in A:
        for word2 in B:
            if word1 == word2:
                rs.append(word1)
    return rs
    
def part6():
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
                
                
                
    p_pos = {}
    p_neg = {}
    for word in list_of_unique_words_train:
        p_pos[word] = (count_pos[word] + m * k)/ (400 * k); # 400 = no. of positive reviews in training set.
        p_neg[word] = (count_neg[word] + m * k)/ (400 * k); # 400 = no. of negatives reviews in training set.
    
    diff = {}
    for word in p_pos:
        diff[word] = p_pos[word] - p_neg[word];
        
    
    print "------------------Naive bayesian-----------------------"
    top50_pos = dict(sorted(diff.iteritems(), key=operator.itemgetter(1), reverse=True)[:50])
    top50_neg = dict(sorted(diff.iteritems(), key=operator.itemgetter(1), reverse=False)[:50])
    
    
    print "Top 50 theta for positive review"
    print top50_pos
    
    print "Top 50 for negative review"
    print top50_neg
    
    
    
    
    
    #-------------------------- LOGISTIC CLASSIFIER----------------------------
    
    # Number of features in a sample
    number_of_features = len(list_of_unique_words_train)
    
    get_index = {}
    i = 0;
    for word in list_of_unique_words_train:
        get_index[word] = i;
        i = i + 1;
        
    
    train_x, train_y = get_data(train_pos, train_neg, list_of_unique_words_train);
    
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
    
    for i in range(200):
        sess.run(train_step, feed_dict={x: train_x, y_: train_y})

    
    train_set_performance = sess.run(accuracy, feed_dict={x: train_x, y_: train_y});
    print train_set_performance
    
    W = sess.run(W0, feed_dict={x: train_x, y_: train_y});
    W_positive = W[:,0]
    W_negative = W[:,1]
    W_diff = W_positive - W_negative;
    
    print W_diff.shape
    
    W_diff_temp = copy(W_diff)
    
    top_50 = {}
    for i in range(50):
        max_index = argmax(W_diff_temp) # get the maximum index.
        word = list_of_unique_words_train[max_index] # word with max theta
        top_50[word] = W_diff_temp[max_index] # stores the word along with the theta.
        W_diff_temp[max_index] = min(W_diff_temp) - 1; # turn to be the lowest value in the array.
        
        
    W_diff_temp = copy(W_diff)
    bottom_50 = {}
    for i in range(50):
        min_index = argmin(W_diff_temp) # get minimum index.
        word = list_of_unique_words_train[min_index] # word with min theta.
        bottom_50[word] = W_diff_temp[min_index]
        W_diff_temp[min_index] = max(W_diff_temp) + 1;
    
    
    print "------------------Logistic Regression-----------------------"
    print "Top 50 theta for positive review"
    print top_50
    
    print "Top 50 theta for negative review"
    print bottom_50
    
    
    print "------------------ Statistic -----------------------"
    #print("Out of 50 words corresponding to the thetas for positive review obtained through Naive Bayesian, {} words are in set of words corresponding to the thetas for negative review obtained through Logistic Regression.")
    
    in_both_pos = words_in_both(top50_pos, top_50)
    in_both_neg = words_in_both(top50_neg, bottom_50)
    
    print("Words appear in both sets of positive reviews.")
    print(in_both_pos)
    print("Num of words in common: {}".format(len(in_both_pos)))
    print("Words appear in both sets of negative reviews.")
    print(in_both_neg)
    print("Num of words in common: {}".format(len(in_both_neg)))
#part6();
    