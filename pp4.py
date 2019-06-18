#!/usr/bin/env python3


###############################################################################
#                                 Import Packages                             #
###############################################################################
import numpy as np
import random
import csv
import matplotlib.pyplot as plt

###############################################################################
#                                 Task 1                                      #
###############################################################################

def readdata():
    print("Reading the data...")
    data=[]
    for i in range(200):
        with open("./pp4data/20newsgroups/"+str(i+1),'r') as f:
            for line in f:
                data.append(line.split())


    with open("./pp4data/20newsgroups/"+"index.csv") as f:
        labels= [int(line.split(',')[1].strip()) for line in f]
       
    artificial=[]
    for i in range(10):
        with open("./pp4data/artificial/"+str(i+1)) as f:
            for line in f:
                artificial.append(line.split())

    with open("./pp4data/artificial/"+"index.csv") as f:
        art_labels=[int(line.split(',')[1].strip()) for line in f]


    return(data, labels, artificial, art_labels)


def pre_gs(data, K):
    # training the model
    w_count_dict={}
    w_i_dict={}
    w_i=[]
    w_sets=[]
    w_indexes=[]
    d_indexes=[]
    z_indexes=[]
    word_index=0
    N_words=0
    D=len(data)

    print("Training the Model...")
    for i in range(len(data)):
        # for each corpus
        for j in range(len(data[i])):
            # for each word
            if w_count_dict.get(data[i][j], 0) > 0:
                w_count_dict[data[i][j]]+=1
                w_indexes.append(w_i_dict[data[i][j]])
                # if word already exists in word dictionary
                

            else:
                # if words doesn't exist in the word dictionary(first time of meeting this word)
                w_sets.append(data[i][j])
                w_i.append(word_index)
                w_i_dict.update({data[i][j]:word_index})
                w_indexes.append(word_index)
                w_count_dict.update({data[i][j]:1})
                word_index+=1
            
            d_indexes.append(i)
            z_indexes.append(np.random.randint(K))
            N_words+=1

            
    return(np.array(w_indexes), np.array(d_indexes), np.array(z_indexes), w_count_dict, w_i_dict, w_i, w_sets, word_index, N_words)


def gibbs_sampling(data, K, Niters):
    random.seed(1)
    w_indexes, d_indexes, z_indexes, w_count_dict, w_i_dict, w_i, w_sets, word_index, N_words= pre_gs(data, K)
    print("Gibbs Sampling... IN PROCESS:   " + str(0) + "   / " + str(Niters) + "  This will take some time.")
    D = len(data)
    alpha = float(5/ K)
    beta = 0.01
    
    # Step 1: Generate a random permutation of the set of N words
    pi_n = np.random.permutation(N_words)
    # Step 2: Initialize a D*K Matrix of topic counts per document Cd (D : # of documents, K: # of topics)
    C_d_mat = np.zeros((D, K))
    

    # Step 3: Initialize a K*V Matrix of word counts per topic Ct (V = # of words in vocabulary)
    V = len(w_sets)
    C_t_mat = np.zeros((K, V))
    for i in range(N_words):
        C_d_mat[d_indexes[pi_n[i]]][z_indexes[pi_n[i]]]+=1
        C_t_mat[z_indexes[pi_n[i]]][w_indexes[pi_n[i]]]+=1
    
    
    # Step 4: Initialize a 1*K array of probabilities P
    prob = np.zeros(K)

    # Step 5: Gibbs Sampling Iteration
    
    # for given iteration
    for i in range(Niters):
        # for every word
        for j in range(N_words):
            # sample word, topic, doc
            word =  w_indexes[pi_n[j]]
            topic = z_indexes[pi_n[j]]
            doc = d_indexes[pi_n[j]]

            # you regard you haven't seen that word, document, topic
            C_d_mat[doc][topic] = C_d_mat[doc][topic] - 1
            C_t_mat[topic][word] = C_t_mat[topic][word] - 1
            
            # for each topic
            for k in range(K):
                topic_sum_per_doc = np.sum(C_d_mat[doc, :])
                word_sum_per_topic= np.sum(C_t_mat[k , :])
                # P(topic) = P(word/topic) * P(topic/doc)
                prob[k] = (C_t_mat[k, word] + beta)/(V * beta + word_sum_per_topic)*(C_d_mat[doc, k]+alpha)/(K*alpha + topic_sum_per_doc)
            
            #  normalize P
            prob_sum = np.sum(prob)
            prob = prob/prob_sum
            rand_prob = np.random.random_sample()
            tmp_prob_sum = 0
            topic = 0
            for k in range(K):
                tmp_prob_sum += prob[k]
                if (rand_prob < tmp_prob_sum):
                    break
                
                else:
                    topic+=1
                    
            z_indexes[pi_n[j]] = topic
            C_d_mat[doc][topic] = C_d_mat[doc][topic]+1
            C_t_mat[topic][word] = C_t_mat[topic][word]+1
        
        if i in (Niters/4, Niters/2, Niters*3/4):
            print("Gibbs Sampling... IN PROCESS:   " + str(i) + " / " + str(Niters))
    
    print("Gibbs Sampling... COMPLETED")
    return(z_indexes, C_d_mat, C_t_mat, w_count_dict, w_sets)

# Input: K, C_t_mat, word_to_indexes | Output: 5 most frequent words
def write_five_frequent(K, C_t_mat, word_to_indexes):
    print("\nOutputting 5 Frequent Words to CSV file...")
    V = len(w_count_dict)
    with open('topicwords.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for k in range(K):
            words_frequencies = []
            for word in range(V):
                words_frequencies.append((C_t_mat[k][word], word_to_indexes[word]))
            words_frequencies = sorted(words_frequencies, reverse= True)[0:5]
            writer.writerow([item[1] for item in words_frequencies])

def get_topic_rep(data, C_d_mat, K):
    print("Getting Topic Representation...")
    alpha = 5 / K
    D=len(data)
    topic_rep=[]
    for doc in range(D):
        tmp_sum_of_doc = np.sum(C_d_mat[doc,:])
        tmp_topic_prob = []
        for topic in range(K):
            prob = (C_d_mat[doc][topic]+alpha) / (K*alpha+tmp_sum_of_doc)
            tmp_topic_prob.append(prob)
        topic_rep.append(tmp_topic_prob)
    
    return(topic_rep)

###############################################################################
#                                 Task 2                                      #
###############################################################################
# Input: Data | Output: probability of words per document
def get_bag_of_words(data):
    print("Getting Bag of Words..")
    D = len(data)
    bag_of_words=[]
    for doc in range(D):
        tmp_word_dic = {keys: 0 for keys in w_sets}
        tmp_word_sum = len(data[doc])
        tmp_word_prob = []
        for word in range(len(data[doc])):
            tmp_word_dic[data[doc][word]] += 1

        for frequency in tmp_word_dic.values():
            prob = frequency / tmp_word_sum
            tmp_word_prob.append(prob) 
            
            
        bag_of_words.append(tmp_word_prob)
              
    return(bag_of_words)

###############################################################################
#                              Code from pp3                                  #
###############################################################################
# I've revised little bit for the code to match with pp4
def sigmoid(a):
    if a>15:
        return(0.99999999)
        
    elif a<-15:
        return(0.00000001)
    
    else:
        return(1.0/(1+np.exp(-a)))

def estimate_class(test_data, w_0, w):
    a = np.dot(test_data, w) + w_0
    prob = [sigmoid(val) for val in a]
    label_est=[]
    for p in prob:   
        if p >= 1/2:
            label_est.append(1)
        else:
            label_est.append(0)
    
    return(label_est)

def split_train_test(phi_mat, label):
    train_data=[]
    train_label=[]
    test_data=[]
    test_label=[]
    
    zipped_data=list(zip(phi_mat,label))
    np.random.shuffle(zipped_data)
    train=zipped_data[len(zipped_data)//3:]
    test=zipped_data[: len(zipped_data)//3]
    
    for j in range(len(train)):
        train_data.append(train[j][0])
        train_label.append(train[j][1])
    
    for i in range(len(test)):
        test_data.append(test[i][0])
        test_label.append(test[i][1])
        
    return(train_data, train_label, test_data, test_label)

def get_accuracy(test_label, est_label):
    # check and calculate the accuracy of the estimation
    correct = 0
    for i in range(len(test_label)):
        if est_label[i]==test_label[i]:
            correct+=1
        
        else:
            continue
        
    return(correct / len(test_label))

def newtons_method(data, label, alpha= 0.01):
    data_arr = np.array(data)
    data_arr = np.insert(data_arr, 0, float(1) ,axis=1)
    features = np.shape(data_arr)[1]
    t = np.array(label)
    
    # Input: w_old | Output: w_new | Calculates a, y R and updates the weights
    def get_w_new(w_old):
        a = np.dot(data_arr, w_old)
        y = np.array([sigmoid(val) for val in a])
        R = np.diag(np.multiply(y, (1-y)))
        first_mat = np.linalg.inv(alpha * np.identity(features) + np.dot(np.dot(data_arr.transpose(), R), data_arr))
        second_mat = np.dot(data_arr.transpose(), (y-t)) + alpha * w_old
        w_new = w_old- np.dot(first_mat, second_mat)
        return(w_new)
    
    w_old= np.zeros((features))
    w_new = get_w_new(w_old)
    count = 0
    w_old[0] = 0.00000001
    while (count == 0 or count<=100) and ((np.linalg.norm(w_new-w_old))/(np.linalg.norm(w_old)) >= 0.001):
        count+=1
        w_old=w_new
        w_new=get_w_new(w_old)
        
    
    w_0=w_new[0]
    w=w_new[1:]
    return(w_0, w)


def get_learning_curve(data, labels, ratio):
    print("Getting Learning Curve...")
    l_curve=[]
    for rep in range(30):
        print("*** Rep " + str(rep+1) + " ***")
        train_data, train_label, test_data, test_label= split_train_test(data, labels)
        tmp_accuracy=[]
        for r in ratio:
            cutline = int(len(train_data)*r)
            samp = train_data[:cutline]
            samp_label = train_label[:cutline]     
            w_0, w = newtons_method(samp, samp_label)
            label_est = estimate_class(test_data, w_0, w)
            tmp_acc=get_accuracy(test_label, label_est)
            tmp_accuracy.append(tmp_acc)
            
        l_curve.append(tmp_accuracy)
    
    print("Getting Learning Curve... complete")
    return(np.array(l_curve))
            


###############################################################################            
#                                 Outputting                                  #
###############################################################################            
print('####################################################################')
print('#                          Task 1                                  #')
print('####################################################################')


# initialize
K = 20
Niters = 500

# read data
data, labels, artificial, art_labels = readdata()

# gibbs sampling
z_indexes, C_d_mat, C_t_mat, w_count_dict, w_sets = gibbs_sampling(data, K, Niters)

# write to csv file
write_five_frequent(K, C_t_mat, w_sets)

# get topic representations
topic_rep = get_topic_rep(data, C_d_mat, K)            

print('####################################################################')
print('#                          Task 2                                  #')
print('####################################################################')

# get bag of words
bag_of_words = get_bag_of_words(data)
        
# initialize r for training data
r =  np.arange(0.05,1.05,0.05)

lc_top, lc_bags=get_learning_curve(topic_rep, labels, r), get_learning_curve(bag_of_words, labels, r)


# plotting!
plt.errorbar(r, np.mean(lc_top, axis=0), yerr=np.std(lc_top, axis=0), ecolor='r',label="LC for Topic Representation")
plt.errorbar(r, np.mean(lc_bags, axis=0), yerr=np.std(lc_bags, axis=0), ecolor='b',label="LC for Bag of Words")
plt.grid()
plt.xlabel("Training Size Ratio")
plt.ylabel("Accuracy")
plt.grid("on")
plt.legend(loc="best")
plt.title("Learning curves for Topic Representation vs Bag of Words")
plt.show()
plt.savefig('task2: learningcurve.png')
