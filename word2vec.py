
from logistic import *
from naivebayes import *

# -----------------------------------------------------------------------------
# PART 7
# -----------------------------------------------------------------------------
# plot learning curve part 7.
def plot_graph_part7(y_test, y_train): 
    y_test = np.asarray(y_test)
    y_train = np.asarray(y_train)

    
    plt.subplot(2,1,1)
    plt.plot(y_test, '-', color='red', lw = 2, label="test set")
    plt.plot(y_train, ':', color='green', lw = 2, label=" train set")
    plt.ylabel('Correctness rate (%)')
    plt.xlabel('Number of iterations')
    plt.title('Learning curves.')
    
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)
           
    plt.show()
    
    
# Picking random number of images for training, testing and validating. 
# The train, test, and valid size are number of documents, NOT the number of pairs of words.
def pick_data7():
    train_size = 15; 
    test_size =  1; 
    valid_size = 1; # Don't need validation for part 7. No paramater needed for the model.
    
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



# w_to_vec = the given matrices of word vectors.
# w_indicies = stores indicies of the corresponding vectors of the words.
# pos_indicies: indicies of the positive review documents to pick the document from.
# neg_indicies: indices of the negative review documents to pick the document from.
def get_data_p7(w_to_vec, w_indicies, pos_indicies, neg_indicies):
    xs = array([])
    ys = array([])
    

    # Building xs and ys for positive reviews.
    for str in sorted(os.listdir(pos_review_dir)):
        if not (int(str[2:5]) in pos_indicies):
            continue
        
        print str
        list_of_words = []; # list of words in a single review.
        
        for line in open(pos_review_dir + str):
            list_of_words += line.split();# array of words in a single line.
        
        for i in range(len(list_of_words)):
            word = list_of_words[i]
            
            # if word appears in the set of Dictionary.
            if word in w_indicies:
                vec_word = w_to_vec[w_indicies[word]] # get vector of the word.
                if (i == 0): # first element in the list.
 
                    if (list_of_words[i + 1] in w_indicies):
                        
                        # add in a pair of words that are NEXT to each other.
                        vec_word_next = w_to_vec[w_indicies[list_of_words[i+1]]];
                        
                        x = hstack((vec_word, vec_word_next)); # 128 * 2
                        
                        if xs.size == 0:
                            xs = vstack([x]);
                        elif xs.size > 0:
                            xs = vstack([xs, x]);
                            
                        one_hot = zeros(2)
                        one_hot[0] = 1 # 10 => words next to each other, 01 means not matched.
                        if ys.size == 0:
                            ys = vstack([one_hot])
                        elif ys.size != 0:
                            ys = vstack([ys, one_hot])   
                        
                        
                        
                        # add in a pair of words that are NOT next to each other.
                        random.seed(int(str[2:5]))
                        word_2 = random.choice(list_of_words);
                        k = 0;
                        while not (word_2 in w_indicies):
                            k = k + 1;
                            random.seed(int(str[2:5]) + k)
                            word_2 = random.choice(list_of_words);
                        
                        vec_x2 = w_to_vec[w_indicies[word_2]]
                        
                        x = hstack((vec_word, vec_x2))
                        
                        xs = vstack([xs, x])
                        one_hot = zeros(2)
                        one_hot[1] = 1
                        ys = vstack([ys, one_hot])
                        
                            
                elif (i == (len(list_of_words) - 1)): # last element in the list.
                    if (list_of_words[i-1] in w_indicies):
                        
                        
                        # add in a pair of words that are NEXT to each other.
                        vec_word_back = w_to_vec[w_indicies[list_of_words[i-1]]]
                        
                        x = hstack((vec_word, vec_word_back)) # 128 * 2
                        
                        if xs.size == 0:
                            xs = vstack([x]);
                        elif xs.size > 0:
                            xs = vstack([xs, x]);
                        
                        
                        one_hot = zeros(2)
                        one_hot[0] = 1 # 10 => words next to each other, 01 means not matched.
                        if ys.size == 0:
                            ys = vstack([one_hot])
                        elif ys.size != 0:
                            ys = vstack([ys, one_hot]) 
                        
                        
                        # add in a pair of words that are NOT next to each other.
                        random.seed(int(str[2:5]))
                        word_2 = random.choice(list_of_words);
                        k = 0;
                        while not (word_2 in w_indicies):
                            k = k + 1;
                            random.seed(int(str[2:5]) + k)
                            word_2 = random.choice(list_of_words);
                        
                        vec_x2 = w_to_vec[w_indicies[word_2]]
                        
                        x = hstack((vec_word, vec_x2))
                        
                        xs = vstack([xs, x])
                        one_hot = zeros(2)
                        one_hot[1] = 1
                        ys = vstack([ys, one_hot])
                        
                        
                else: # in between
                    
                    if (list_of_words[i - 1] in w_indicies):
                        vec_word_back = w_to_vec[w_indicies[list_of_words[i-1]]]
                        
                        x = hstack((vec_word, vec_word_back)) # 128 * 2
                        
                        if xs.size == 0:
                            xs = vstack([x]);
                        elif xs.size > 0:
                            xs = vstack([xs, x]);
                            
                        one_hot = zeros(2)
                        one_hot[0] = 1 # 10 => words next to each other, 01 means not matched.
                        if ys.size == 0:
                            ys = vstack([one_hot])
                        elif ys.size != 0:
                            ys = vstack([ys, one_hot]) 
                    
                    
                        random.seed(int(str[2:5]))
                        word_2 = random.choice(list_of_words);
                        k = 0;
                        while not (word_2 in w_indicies):
                            k = k + 1;
                            random.seed(int(str[2:5]) + k)
                            word_2 = random.choice(list_of_words);
                        
                        vec_x2 = w_to_vec[w_indicies[word_2]]
                        
                        x = hstack((vec_word, vec_x2))
                        
                        xs = vstack([xs, x])
                        one_hot = zeros(2)
                        one_hot[1] = 1
                        ys = vstack([ys, one_hot])
                        
                    
                    if (list_of_words[i + 1] in w_indicies):
                        vec_word_next = w_to_vec[w_indicies[list_of_words[i+1]]];
                        
                        x = hstack((vec_word, vec_word_next)); # 128 * 2
                        
                        if xs.size == 0:
                            xs = vstack([x]);
                        elif xs.size > 0:
                            xs = vstack([xs, x]);
                        
                        
                        one_hot = zeros(2)
                        one_hot[0] = 1 # 10 => words next to each other, 01 means not matched.
                        if ys.size == 0:
                            ys = vstack([one_hot])
                        elif ys.size != 0:
                            ys = vstack([ys, one_hot])
        
                        random.seed(int(str[2:5]))
                        word_2 = random.choice(list_of_words);
                        k = 0;
                        while not (word_2 in w_indicies):
                            k = k + 1;
                            random.seed(int(str[2:5]) + k)
                            word_2 = random.choice(list_of_words);
                        
                        vec_x2 = w_to_vec[w_indicies[word_2]]
                        
                        x = hstack((vec_word, vec_x2))
                        
                        xs = vstack([xs, x])
                        one_hot = zeros(2)
                        one_hot[1] = 1
                        ys = vstack([ys, one_hot])
    
    
    # ----------------------------------------------------------------
    # Repeat what we did for positive reviews on negative reviews.
    # ----------------------------------------------------------------
    for str in sorted(os.listdir(neg_review_dir)):

        if not (int(str[2:5]) in neg_indicies):
            continue
        print str
        list_of_words = []; # list of words in a single review.
        for line in open(neg_review_dir + str):
            list_of_words += line.split();# array of words in a single line.
        
        for i in range(len(list_of_words)):
            word = list_of_words[i]
            
            # if word appears in the set of Dictionary.
            if word in w_indicies:
                vec_word = w_to_vec[w_indicies[word]] # get vector of the word.
                if (i == 0): # first element in the list.
 
                    if (list_of_words[i + 1] in w_indicies):
                        vec_word_next = w_to_vec[w_indicies[list_of_words[i+1]]];
                        
                        x = hstack((vec_word, vec_word_next)); # 128 * 2
                        
                        if xs.size == 0:
                            xs = vstack([x]);
                        elif xs.size > 0:
                            xs = vstack([xs, x]);
                            
                        one_hot = zeros(2)
                        one_hot[0] = 1 # 10 => words next to each other, 01 means not matched.
                        if ys.size == 0:
                            ys = vstack([one_hot])
                        elif ys.size != 0:
                            ys = vstack([ys, one_hot])   
                        
                        
                        random.seed(int(str[2:5]))
                        word_2 = random.choice(list_of_words);
                        k = 0;
                        while not (word_2 in w_indicies):
                            k = k + 1;
                            random.seed(int(str[2:5]) + k)
                            word_2 = random.choice(list_of_words);
                        
                        vec_x2 = w_to_vec[w_indicies[word_2]]
                        
                        x = hstack((vec_word, vec_x2))
                        
                        xs = vstack([xs, x])
                        one_hot = zeros(2)
                        one_hot[1] = 1
                        ys = vstack([ys, one_hot])
                        
                elif (i == (len(list_of_words) - 1)): # last element in the list.
                    if (list_of_words[i-1] in w_indicies):
                        vec_word_back = w_to_vec[w_indicies[list_of_words[i-1]]]
                        
                        x = hstack((vec_word, vec_word_back)) # 128 * 2
                        
                        if xs.size == 0:
                            xs = vstack([x]);
                        elif xs.size > 0:
                            xs = vstack([xs, x]);
                        
                        
                        one_hot = zeros(2)
                        one_hot[0] = 1 # 10 => words next to each other, 01 means not matched.
                        if ys.size == 0:
                            ys = vstack([one_hot])
                        elif ys.size != 0:
                            ys = vstack([ys, one_hot]) 
                        
            
                        random.seed(int(str[2:5]))
                        word_2 = random.choice(list_of_words);
                        k = 0;
                        while not (word_2 in w_indicies):
                            k = k + 1;
                            random.seed(int(str[2:5]) + k)
                            word_2 = random.choice(list_of_words);
                        
                        vec_x2 = w_to_vec[w_indicies[word_2]]
                        
                        x = hstack((vec_word, vec_x2))
                        
                        xs = vstack([xs, x])
                        one_hot = zeros(2)
                        one_hot[1] = 1
                        ys = vstack([ys, one_hot])
                        
                else: # in between
                    
                    if (list_of_words[i - 1] in w_indicies):
                        vec_word_back = w_to_vec[w_indicies[list_of_words[i-1]]]
                        
                        x = hstack((vec_word, vec_word_back)) # 128 * 2
                        
                        if xs.size == 0:
                            xs = vstack([x]);
                        elif xs.size > 0:
                            xs = vstack([xs, x]);
                            
                        one_hot = zeros(2)
                        one_hot[0] = 1 # 10 => words next to each other, 01 means not matched.
                        if ys.size == 0:
                            ys = vstack([one_hot])
                        elif ys.size != 0:
                            ys = vstack([ys, one_hot]) 
                        
                        
                        random.seed(int(str[2:5]))
                        word_2 = random.choice(list_of_words);
                        k = 0;
                        while not (word_2 in w_indicies):
                            k = k + 1;
                            random.seed(int(str[2:5]) + k)
                            word_2 = random.choice(list_of_words);
                        
                        vec_x2 = w_to_vec[w_indicies[word_2]]
                        
                        x = hstack((vec_word, vec_x2))
                        
                        xs = vstack([xs, x])
                        one_hot = zeros(2)
                        one_hot[1] = 1
                        ys = vstack([ys, one_hot])
                    
                    if (list_of_words[i + 1] in w_indicies):
                        vec_word_next = w_to_vec[w_indicies[list_of_words[i+1]]];
                        
                        x = hstack((vec_word, vec_word_next)); # 128 * 2
                        
                        if xs.size == 0:
                            xs = vstack([x]);
                        elif xs.size > 0:
                            xs = vstack([xs, x]);
                        
                        
                        one_hot = zeros(2)
                        one_hot[0] = 1 # 10 => words next to each other, 01 means not matched.
                        if ys.size == 0:
                            ys = vstack([one_hot])
                        elif ys.size != 0:
                            ys = vstack([ys, one_hot])
                        
                        
                        random.seed(int(str[2:5]))
                        word_2 = random.choice(list_of_words);
                        k = 0;
                        while not (word_2 in w_indicies):
                            k = k + 1;
                            random.seed(int(str[2:5]) + k)
                            word_2 = random.choice(list_of_words);
                        
                        vec_x2 = w_to_vec[w_indicies[word_2]]
                        
                        x = hstack((vec_word, vec_x2))
                        
                        xs = vstack([xs, x])
                        one_hot = zeros(2)
                        one_hot[1] = 1
                        ys = vstack([ys, one_hot])
    
    return xs, ys
                    
            
                
                
            

def part7():
    
    ll = pick_data7();

    train_pos = ll[0];
    train_neg = ll[3];
    
    test_pos = ll[2];
    test_neg = ll[5];
    
    vec = load("embeddings.npz")["emb"]
    w_indicies = load("embeddings.npz")["word2ind"].flatten()[0]
    w_indicies = dict((v, k) for k,v in w_indicies.iteritems())
    
    train_x, train_y = get_data_p7(vec, w_indicies, train_pos, train_neg)
    test_x, test_y = get_data_p7(vec, w_indicies, test_pos, test_neg)
    
    
    #-----------------Tensorflow logistic model-------------------------------
    x = tf.placeholder(tf.float32, [None, 256])
    
    
    W0 = tf.Variable(tf.random_normal([256, 2], stddev=0.01))
    b0 = tf.Variable(tf.random_normal([2], stddev=0.01))
    
    layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)

    y = tf.nn.softmax(layer1) # computed result
    y_ = tf.placeholder(tf.float32, [None, 2]) # expect result
    
    lam = 0
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

    for i in range(100):
        sess.run(train_step, feed_dict={x: train_x, y_: train_y})
        
        train_set_performance = sess.run(accuracy, feed_dict={x: train_x, y_: train_y});
        test_set_performance = sess.run(accuracy, feed_dict={x: test_x, y_: test_y});
        
        print("i=", i)
        print("Performance on the test set: ", test_set_performance)
        print("Performance on the train set: ", train_set_performance)

        y_test.append(test_set_performance * 100)
        y_train.append(train_set_performance * 100)
        

    print "Number of pairs of words in the training set"
    print train_x.shape[0]
    print "Number of pairs of words in the test set"
    print test_x.shape[0]
    plot_graph_part7(y_test, y_train);
#part7()






# -----------------------------------------------------------------------------
# PART 8
# -----------------------------------------------------------------------------
def part8():
    
    vec = load("embeddings.npz")["emb"]
    w_key = load("embeddings.npz")["word2ind"].flatten()[0]
    w_indicies = dict((v, k) for k,v in w_key.iteritems()) # key to value, value to key
    
    
    # 10 words whose embedding are closest to the embedding of "good".
    vector_good = vec[w_indicies["good"]]
    smallest = zeros(10);
    smallest_indicies = zeros(10); # stores corresponding indices of the words.
    curr_len = 0;
    for i in range(vec.shape[0]):
        if i == w_indicies["good"]: # avoid matching to the word itself.
            continue
            
        eclu_distance = (sum((vector_good - vec[i]) ** 2)) ** 0.5
        
        # get smallest 10.
        if curr_len != 10:
            smallest[curr_len] = eclu_distance;
            curr_len += 1;
        elif curr_len == 10:
            max_index = argmax(smallest)
            if (eclu_distance < smallest[max_index]):
                smallest[max_index] = eclu_distance;
                smallest_indicies[max_index] = i;
    
    print smallest_indicies
    
    print "10 words that are related to good"
    for index in smallest_indicies:
        print w_key[index]
    
    
    # 10 words whose embedding are closest to the embedding of "story".
    vector_story = vec[w_indicies["story"]]
    smallest = zeros(10);
    smallest_indicies = zeros(10); # stores corresponding indices of the words.
    curr_len = 0;
    for i in range(vec.shape[0]):
        if i == w_indicies["story"]: # avoid matching to the word itself.
            continue
        
        eclu_distance = (sum((vector_story - vec[i]) ** 2)) ** 0.5
        
        
        if curr_len != 10:
            smallest[curr_len] = eclu_distance;
            curr_len += 1;
        elif curr_len == 10:
            max_index = argmax(smallest)
            if (eclu_distance < smallest[max_index]):
                smallest[max_index] = eclu_distance;
                smallest_indicies[max_index] = i;
    
    print "10 words that are related to story"
    for index in smallest_indicies:
        print w_key[index]
    
    
    
    
    # 2 more examples
    
    # 10 words whose embedding are closest to the embedding of "very".
    vector_very = vec[w_indicies["very"]]
    smallest = zeros(10);
    smallest_indicies = zeros(10); # stores corresponding indices of the words.
    curr_len = 0;
    for i in range(vec.shape[0]):
        if i == w_indicies["very"]: # avoid matching to the word itself.
            continue
        
        eclu_distance = (sum((vector_very - vec[i]) ** 2)) ** 0.5
        
        if curr_len != 10:
            smallest[curr_len] = eclu_distance;
            curr_len += 1;
        elif curr_len == 10:
            max_index = argmax(smallest)
            if (eclu_distance < smallest[max_index]):
                smallest[max_index] = eclu_distance;
                smallest_indicies[max_index] = i;
    
    
    
    print "10 words that are related to very"
    for index in smallest_indicies:
        print w_key[index]
    
    
    # 10 words whose embedding are closest to the embedding of "best".
    vector_best = vec[w_indicies["best"]]
    smallest = zeros(10);
    smallest_indicies = zeros(10); # stores corresponding indices of the words.
    curr_len = 0;
    for i in range(vec.shape[0]):
        if i == w_indicies["best"]: # avoid matching to the word itself.
            continue
            
        eclu_distance = (sum((vector_best - vec[i]) ** 2)) ** 0.5
        
        if curr_len != 10:
            smallest[curr_len] = eclu_distance;
            curr_len += 1;
        elif curr_len == 10:
            max_index = argmax(smallest)
            if (eclu_distance < smallest[max_index]):
                smallest[max_index] = eclu_distance;
                smallest_indicies[max_index] = i;
    
    
    
    print "10 words that are related to best"
    for index in smallest_indicies:
        print w_key[index]
#part8();