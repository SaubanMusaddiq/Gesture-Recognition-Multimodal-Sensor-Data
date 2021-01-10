import pandas as pd
import scipy.integrate as integrate
import numpy as np
from itertools import accumulate
import matplotlib.pyplot as plt

from collections import defaultdict
import os
import glob
import math
from os import listdir
import seaborn as sns

import random
import json
import webview

import argparse

import threading
import time
import sys


def get_no_of_sensors(dr):
    with open(os.path.join(dr, [f for f in os.listdir(dr) if f.endswith('.csv')][1]), 'r') as file:
        no_of_sensors = len(file.readlines())
    return no_of_sensors

def get_all_gesture_ids(dr):
    onlyfiles = [f for f in listdir(dr) if os.path.isfile(os.path.join(dr, f))]
    if(len(onlyfiles) == 0):
        print("Task 0 not completed: Create Wrd files")
        return
    gesture_files = [f.split(".")[0] for f in onlyfiles]
    gesture_files.sort()
    return gesture_files


# Creating gesture Words


def get_gaussian_quantization_bins(r = 3):
    den = integrate.quad(lambda x: np.exp(-8*x**2) , -1,1)
    nums = [integrate.quad(lambda x: np.exp(-8*x**2) , (i-r)/r,(i-r+1)/r) for i in range(2*r)]
    bands = [(2* nums[i][0]-nums[i][1])/(den[0]-den[1]) for i in range(2*r)]
    return [i-1 for i in accumulate(bands)]

def create_gesture_words(fdata, comp_name, w, s, r):
    # Normalize
    fdata=(fdata-fdata.min())/(fdata.max()-fdata.min()) * 2 - 1

    # get gaussian quantization bins
    quantization_bins = get_gaussian_quantization_bins(r)
    # Quantize into 2*r ranges
    fdata = fdata.apply(lambda x: np.digitize(x, bins = quantization_bins ))
    file_gesture_words = []
    # For each column(sensor data) get the words
    for sensor_id in range(fdata.shape[1]):
        sensor_data = fdata.iloc[:,sensor_id].tolist()
        sensor_word_count = int(len(sensor_data)/s)
        sensor_gesture_words = [(i*s, sensor_data[i*s:i*s+w]) for i in range(sensor_word_count) if i*s+w <= len(sensor_data)]
        # Sensor id +1 to offset the 0 indexed sensor id
        file_gesture_words += [(comp_name, sensor_id+1,tuple(word)) for (t,word) in sensor_gesture_words]

    return file_gesture_words

def create_gesture_wordfiles_of_component(dr, comp_name,  w = 3, s = 3, r = 3):
    comp_dr = dr+comp_name
    files = [f for f in os.listdir(comp_dr) if f.endswith('.csv')]
    no_of_gestures = len(files)
    fmode = "w" if comp_name == "X" else "a"
    for f in files:
        # read and Transpose
        fdata = pd.read_csv(os.path.join(comp_dr, f), header = None).T
        f = f.split(".")[0]
        file_gesture_words = create_gesture_words(fdata, comp_name, w, r, s)
        with open(os.path.join(dr+"gesture_words", f.split(".")[0]+".wrd"), fmode) as file:
            for item in file_gesture_words:
                file.write("%s\n" % str(item))

def create_gesture_wordfiles(dr,w,s,r):
    if not os.path.exists(dr+"gesture_words"):
        os.makedirs(dr+"gesture_words")

    for comp in ["X","Y","Z","W"]:
        create_gesture_wordfiles_of_component(dr,comp,w,s,r)

# creating  gesture vecors

# Task 0b
def update_word_doc_freq(word_tuples, word_doc_freq):
    seen = []
    prev_sensor_id = (0,0)
    for word_tuple in word_tuples:
        sensor_id = (word_tuple[0],word_tuple[1])
        if(sensor_id != prev_sensor_id):
            seen = []
        prev_sensor_id = sensor_id
        word = word_tuple[2]
        if(word not in seen):
            if word in word_doc_freq[sensor_id]:
                word_doc_freq[sensor_id][word] += 1
            else:
                word_doc_freq[sensor_id][word] = 1
            seen.append(word)



def create_word_doc_frequencies_for_sensor(dr, no_of_sensors, comps):
    word_doc_freq = {}
    for c in comps:
        for i in range(1,no_of_sensors +1):
            word_doc_freq[(c,i)] = {}

    files = (f for f in os.listdir(dr) if f.endswith('.wrd'))
    for f in files:
        with open(os.path.join(dr, f), 'r') as file:
            word_tuples = list(map(lambda x: eval(x),file.readlines()))
        update_word_doc_freq(word_tuples, word_doc_freq)
    return word_doc_freq



def get_word_frequencies_in_gesture(word_tuples):
    word_freq =  {}
    for word_tuple in word_tuples:
        if word_tuple in word_freq:
            word_freq[word_tuple] += 1
        else:
            word_freq[word_tuple] = 1
    return word_freq


def update_gesture_vectors_per_gesture(word_tuples, gesture_id, gesture_vectors, word_doc_freq, no_of_gestures = 60, no_of_sensors = 20):
    tf = gesture_vectors["tf"]
    tfidf = gesture_vectors["tfidf"]
    tf[gesture_id] = {}
    tfidf[gesture_id]  = {}

    word_freq =  get_word_frequencies_in_gesture(word_tuples)
    word_count = len(word_tuples)/(no_of_sensors*4)
#     word_count = len(word_tuples)

    for (tup , n) in word_freq.items():
        sensor_id = (tup[0],tup[1])
        word = tup[2]
        term_freq =  n / word_count
        idf = math.log(no_of_gestures / word_doc_freq[sensor_id][word])

        tf[gesture_id][tup] = term_freq
        tfidf[gesture_id][tup] = term_freq * idf


def create_gesture_vectors(dr, no_of_sensors = 20, comps= ["X","Y","Z","W"]):
    tf = {}
    tfidf = {}
    gesture_vectors = {
        "tf" : tf,
        "tfidf" : tfidf
    }
    wrd_dr = dr+"gesture_words/"

    files = [f for f in os.listdir(wrd_dr) if f.endswith('.wrd')]
    no_of_gestures = len(files)

    if(no_of_gestures == 0):
        print("Error: Run task0-a first")
        return

    word_doc_freq = create_word_doc_frequencies_for_sensor(wrd_dr, no_of_sensors, comps)

    for f in files:
        gesture_id = f.split(".")[0]
        with open(os.path.join(wrd_dr, f), 'r') as file:
            word_tuples = list(map(lambda x: eval(x),file.readlines()))
        update_gesture_vectors_per_gesture(word_tuples, gesture_id, gesture_vectors, word_doc_freq, no_of_gestures, no_of_sensors)
    with open(os.path.join(dr+"vectors/", "tf_vectors.txt"), 'w') as file:
        file.write(str(gesture_vectors["tf"]))
    with open(os.path.join(dr+"vectors/", "tfidf_vectors.txt"), 'w') as file:
        file.write(str(gesture_vectors["tfidf"]))

    print("Created the word vectors and stored in "+dr+"vectors dir")

# Create data Mat

def get_words_with_index_map(vect, gesture_files):
    uwords = set()
    for gid in gesture_files:
        uwords.update(vect[gid].keys())
    uwords = list(uwords)
    uwords.sort()
    mp = {k: v for v, k in enumerate(uwords)}
    mpi = {v: k for v, k in enumerate(uwords)}
    return (mp, mpi)

def create_data_mat(dr, vector_type, gesture_files, no_of_gestures, no_of_sensors):
    if(vector_type== "tf"):
        path = os.path.join(dr+"vectors/", "tf_vectors.txt")
    else:
        path = os.path.join(dr+"vectors/", "tfidf_vectors.txt")
    with open(path, 'r') as file:
        vect =  eval(file.read())
    word_index_mp, index_word_mp = get_words_with_index_map(vect,gesture_files)
    no_of_features = len(word_index_mp)
    data_mat = np.zeros((no_of_gestures,no_of_features))
    g_i = 0
    for gid in gesture_files:
        for (key,val) in vect[gid].items():
            data_mat[g_i][word_index_mp[key]] = val
        g_i +=1
    return (data_mat, index_word_mp)



# Creating Gesture similarity Matrix


def edit_dist_cost(a, b):
    return sum([abs(i-j) for i,j in zip(a,b)])

def edit_distance(seq1, seq2, r=3):
    l1 = len(seq1) + 1
    l2 = len(seq2) + 1
    memo = [[0] * l1 for _ in range(l2)]
    mp = tuple([r - 0.5] * len(seq1[0])) # Agrregate symbol, Eg, when r =3, w= 3, mp = (2.5,2.5,2.5)
    for j in range(1,l1):
        memo[0][j] = memo[0][j-1] + edit_dist_cost(seq1[j-1], mp)
    for i in range(1,l2):
        memo[i][0] = memo[i-1][0] + edit_dist_cost(seq2[i-1], mp)
    for i in range(1,l2):
        for j in range(1,l1):
            rc = edit_dist_cost(seq1[j-1], seq2[i-1]) + memo[i-1][j-1]
            dc = edit_dist_cost(seq1[j-1], mp) + memo[i][j-1]
            ic = edit_dist_cost(seq2[i-1], mp) + memo[i-1][j]
            memo[i][j] =  min(rc,dc,ic)
    return memo[-1][-1]

def get_gesture_edit_distances(dr, gesture_id, r):
    with open(os.path.join(dr+"gesture_words/", gesture_id+".wrd"), 'r') as file:
        word_tuples1 = list(map(lambda x: eval(x),file.readlines()))
    no_of_sensors =  get_no_of_sensors(dr+"X")
    sensor_words_map1 = defaultdict(list)
    for word in word_tuples1:
        sensor_words_map1[(word[0],word[1])].append(word[2])
    gesture_dists = []
    gesture_ids = get_all_gesture_ids(dr+"gesture_words/")
    for gesture_id2 in gesture_ids:
        with open(os.path.join(dr+"gesture_words/", gesture_id2+".wrd"), 'r') as file:
            word_tuples2 = list(map(lambda x: eval(x),file.readlines()))
        sensor_words_map2 = defaultdict(list)
        for word in word_tuples2:
            sensor_words_map2[(word[0],word[1])].append(word[2])
        sensor_dists = []
        for comp in ["W","X","Y","Z"]:
            for sid in range(1,no_of_sensors+1):
                seq1 = sensor_words_map1[(comp,sid)]
                seq2 = sensor_words_map2[(comp,sid)]
                sensor_dists.append(edit_distance(seq1, seq2, r))
        gesture_dists.append(sum(sensor_dists))
    return gesture_dists

def get_gest_mat_edit_distance(dr, gesture_files, r):
    out_mat = []
    for file_id in gesture_files:
        gest_dist = get_gesture_edit_distances(dr, file_id, r)
        out_mat.append(gest_dist)
    gesture_mat = np.array(out_mat)
    return gesture_mat

# create sim matrix

def create_sim_matrix(dr, r, tp):
    if(tp == "dot"):
        with open(os.path.join(dr+"meta_data/", "data_mat.npy"), 'rb') as f:
            data_mat = np.load(f)
        sim_mat = np.dot(data_mat,data_mat.T)
        return sim_mat
    else:
        gesture_files = get_all_gesture_ids(dr+"gesture_words/")
        in_mat = get_gest_mat_edit_distance(dr, gesture_files, r)
        mn = np.min(in_mat[np.nonzero(in_mat)]) * 0.9
        mx = np.max(in_mat[np.nonzero(in_mat)])
        sim_mat = np.minimum(np.ones(in_mat.shape), np.divide((mx - in_mat),( mx - mn)))
        return sim_mat

def get_sim_matrix(dr, r=3, tp = "dot"):
    if(tp == "dot"):
        sim_mat_file = os.path.join(dr+"meta_data/", "sim_mat_dt.csv")
    else:
        sim_mat_file = os.path.join(dr+"meta_data/", "sim_mat_ed.csv")
    if(os.path.exists(sim_mat_file)):
        print("Loaded similarity matrix from : ", sim_mat_file)
        sim_mat = np.genfromtxt(sim_mat_file, delimiter=',')
    else:
        sim_mat =  create_sim_matrix(dr, r, tp)
        print("Computed similarity matrix, Saved: ",sim_mat_file)
        with open(sim_mat_file, 'w') as f:
            np.savetxt(f, sim_mat, delimiter=",")
    return sim_mat

# task 0

def preprocessing_tasks(dr, w, s, r, vector_type):
    create_gesture_wordfiles(dr, w, s, r)
    gesture_files = get_all_gesture_ids(dr+"gesture_words/")
    gesture_index = {ky: v for v, ky in enumerate(gesture_files)}
    with open(os.path.join(dr+"meta_data/", "gesture_index.txt"), 'w') as file:
        file.write(str(gesture_index))

    no_of_sensors =  get_no_of_sensors(dr+"X")
    no_of_gestures = len(gesture_files)
    create_gesture_vectors(dr, no_of_sensors = 20, comps = ["X","Y","Z","W"])
    data_mat, index_word_mp = create_data_mat(dr, vector_type, gesture_files, no_of_gestures, no_of_sensors)
    with open(os.path.join(dr+"meta_data/", "data_mat.npy"), 'wb') as f:
        np.save(f, data_mat)



# # Task 1: Get Representative Gestures using PPR algorithm

def normalize_sum(arr):
    return arr/arr.sum(axis=0,keepdims=1)

def get_tran_mat(sim_mat, k):
    if( k >= sim_mat.shape[0] - 1):
        print("Error: K should be less than ", sim_mat.shape[0]-1)
        return
    tran_mat = np.zeros(sim_mat.shape)
    for i in range(sim_mat.shape[0]):
        topk_idx = np.argsort(sim_mat[i])[-k-1:-1][::-1]
        tran_mat[i][topk_idx] = normalize_sum(sim_mat[i][topk_idx])
    return tran_mat

def get_representative_gestures(dr, tran_mat, sel_gestures_list, m, c= 0.15):
    with open(os.path.join(dr+"meta_data/", "gesture_index.txt"), 'r') as file:
        gesture_index_map = eval(file.read())
    try:
        sel_gestures_idx = [gesture_index_map[i] for i in sel_gestures_list]
    except:
        print("One or more of the gestures are invalid")
        return
    index_gesture_map = {v:k for k,v in gesture_index_map.items()}
    seed_vector = np.zeros((tran_mat.shape[0],1))
    seed_vector[sel_gestures_idx] = 1.0/len(sel_gestures_list)
    ppr_vector = np.copy(seed_vector)
    for i in range(100):
        last_vector = np.copy(ppr_vector)
        ppr_vector = np.matmul(tran_mat, ppr_vector) * (1-c) + seed_vector * c
        if(np.allclose(ppr_vector, last_vector, rtol = 1e-4)):
            print("PPR algorithm converged after "+ str(i+1) +" iterations")
            break
    else:
        print("PPR algorithm didnt converge in 100 iterations")
    max_m_idx = np.argsort(ppr_vector.T)[0,-m:][::-1]
    max_m_gesture = [index_gesture_map[i] for i in list(max_m_idx)]
    print("{} most dominant gestures using PPR algo: {}".format(m, " ".join(max_m_gesture)))
    return max_m_gesture


def visualize_gestures(dr, gestures):
    files = glob.glob(dr+"gesture_images/*")
    # plt.ioff()
    for f in files:
        os.remove(f)
    for gest in gestures:
        rolling_mean_len = 7
        fig, axs = plt.subplots(nrows =2, ncols=2, figsize=(18, 13))
        for i,j,comp in [(0,0,"W"),(0,1,"X"),(1,0,"Y"),(1,1,"Z")]:
            fdata = pd.read_csv(os.path.join(dr+comp, gest+'.csv'), header = None).T
            fdata =(fdata-fdata.min())/(fdata.max()-fdata.min()) * 2 - 1
            fdata = fdata.rolling(rolling_mean_len).mean().iloc[rolling_mean_len-1:].reset_index()
            fdata = fdata.drop(columns=['index'])
            fdata["t"] = list(range(fdata.shape[0]))
            sns.lineplot(x='t', y='value', hue='Sensors',
                         data=pd.melt(fdata, ['t'], var_name='Sensors'),  ax=axs[i,j],
                         palette=sns.color_palette("dark",20))
            if(j==0):
                axs[i,j].legend([],[], frameon=False)
            else:
                axs[i,j].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            axs[i,j].set_ylabel(comp)
        #     axs[i,j].annotate(0, xy=(fdata.shape[0],fdata.iloc[-1,0]), xytext=(fdata.shape[0]-1,fdata.iloc[-1,0]))
        #     for k in range(20):
        #         axs[i,j].text(fdata.shape[0], fdata.iloc[-1,k], k, ha='center', va='center', fontsize=7)
        fig.savefig(dr+"gesture_images/"+gest+'.png')
    print("Saved the gesture visualizations in gesture_images")



# # Task 2


def create_tran_mat(sim_mat, vector):
    tran_mat = np.zeros(sim_mat.shape)

    for i in vector:
        for j in vector:
            tran_mat[i][j] = sim_mat[i][j]
        tran_mat[i] = normalize_sum(tran_mat[i])
    #print(tran_mat)
    return tran_mat



#Task 2: Create a PPR based classifier
# 1. Read the training labels from sample_training_labels.xlsx
# 2. For each of the remaining gestures, label them by considering it as the seed node.
# Then using the 30+1 nodes as nodes of interest find the PPR and mark the seed node based on the values of the dominant gestures.
# k = 31
# m = 10

def PPR_classify(dr, sample_file, sim_mat):
    sample_training = pd.read_excel(dr+sample_file, index_col = None, header = None)
    with open(os.path.join(dr+"meta_data/", "gesture_index.txt"), 'r') as file:
        gesture_index_map = eval(file.read())
    index_gesture_map = {v:k for k,v in gesture_index_map.items()}

    all_labels = []
    #tran_mat = get_tran_mat(sim_mat, 30)
    index = [str(i) for i in sample_training[0]]

    ges_label = []
    for i in range(len(index_gesture_map)):

        if(int(index_gesture_map[i]) not in sample_training.values):
            print("gesture in check: ", index_gesture_map[i])
            ges_label = ges_label.clear()
            ges_label = [gesture_index_map[x] for x in index]
            ges_label.append(i)
            tran_mat = create_tran_mat(sim_mat, ges_label) #gives importance based on vectors of choice.

            max_m_gesture = get_representative_gestures(dr, tran_mat, [index_gesture_map[i]], 6 , 0.15)
            gesture = list(map(int, max_m_gesture))
            match_labels = sample_training.loc[(sample_training[0].isin(gesture))].groupby([1]).count()#.reset_index()]

            label_len = len(all_labels)
            if not match_labels.empty:
                #print([index_gesture_map[i],match_labels.iloc[0][1]])
                all_labels.append([index_gesture_map[i],match_labels.idxmax()[0]])
            else:
                all_labels.append([index_gesture_map[i],''])

    df = pd.DataFrame(all_labels, columns = ["gesture_id","label"])
    df.to_csv(dr+"meta_data/label_data.csv")


    check = pd.read_excel(dr+'labels.xlsx', index_col = None, header = None)
    check = check.rename(columns={0:"gesture_id", 1:"label"})
    df = df.append(sample_training.rename(columns ={0:"gesture_id", 1:"label"}), ignore_index=True)
    # df["gesture_id"] = df["gesture_id"].astype(int)

    match = df.merge(check, on=["gesture_id", "label"], how = 'inner')

    print("Accuracy: ", len(match)/len(df))
    #match = np.where(df == check)
    #unmatch = np.where(df != check)
    #print(len(unmatch[0]))

    return

# Task 5


def relevance_system(dr, sim_mat, gesture, m, k, c):
    no_of_loop = 0

    tran_mat = get_tran_mat(sim_mat, k) #k = 10
    new_gestures = []
    while True:
        check = True
        sim_gestures = get_representative_gestures(dr, tran_mat, gesture, m, c)
        print(len(sim_gestures))
        print("The ", m, " most similar gestures are \n", sim_gestures)
        if(gesture == sim_gestures):
            print("The set recived is the same as previous.")
        stop = input("Are you satisfied?(Y/n)")
        if((stop == 'Y')or(stop == 'y')):
            do = False
            break
        while check:
            check = False
            print("Enter a list of 1 or -1; \n '1' if the gesture is relevant and \n'-1' is it is irrelevant")
            print("\n\n:NOTE: The input will be a list of ",m," entries separated by ',' ")
            rel_str = input()
            rel = rel_str.split(',')
            rel_gestures = [r for r in rel if r != '']
            try:
                rel_gestures = [int(i) for i in rel_gestures]
            except:
                print("Non-integer provided. input error, Please check")
                check = True
            if(len(rel_gestures) != m):
                check = True
            for r in rel_gestures:
                if((r != int(1)) and (r != int(-1))):
                    check = True
                    break

        if(all(r == 1 for r in rel_gestures)):
            print(" You have marked all the gestures as relevant. Quitting the program")
            break
        for i in range(len(rel_gestures)-1):
            if(rel_gestures[i] == 1):
                new_gestures.append(sim_gestures[i])
        gesture = new_gestures

    return



class GestureApi:
    def __init__(self, dr, r):
        self.sim_mat = get_sim_matrix(dr, r)
        self.k = 10
        self.tran_mat = get_tran_mat(self.sim_mat, self.k) #k = 10
        self.m = 10
        self.c = 0.15
        self.gestures = []
        self.sim_gestures = []
        self.dr = dr

    def searchGestures(self, gestures):
        if(type(gestures) == str):
            gestures = gestures.split(",")
        print(gestures)
        sim_gestures = get_representative_gestures(self.dr, self.tran_mat, gestures, self.m, self.c)
        self.gestures = gestures
        self.sim_gestures = sim_gestures
        response = {
            'list': sim_gestures
        }

        return response

    def updateSearch(self, rel_gestures):
        gestures = []
        for i in range(len(rel_gestures)):
            if(rel_gestures[i] == 1):
                gestures.append(self.sim_gestures[i])

        return self.searchGestures(gestures)


    def error(self):
        raise Exception('This is a Python exception')




def main(config):
    if not os.path.exists(config.dr+"meta_data"):
        os.makedirs(config.dr+"meta_data")
    if not os.path.exists(config.dr+"gesture_images"):
        os.makedirs(config.dr+"gesture_images")
    if not os.path.exists(config.dr+"vectors"):
        os.makedirs(config.dr+"vectors")

    if config.task == "task0":
        preprocessing_tasks(config.dr, config.w, config.s, config.r, config.vect_type)
        sim_mat = get_sim_matrix(config.dr, config.r, config.mat_type)
        print("Task 0 done")
    elif config.task == "task1":
        sim_mat = get_sim_matrix(config.dr,  config.r, config.mat_type)
        tran_mat = get_tran_mat(sim_mat, config.k)

        gestures_list = config.q.split(",")
        max_m_gesture = get_representative_gestures(config.dr, tran_mat, gestures_list, config.m , config.c )
        # visualize_gestures(config.dr, max_m_gesture)
    elif config.task == "task2":
        sim_mat = get_sim_matrix(config.dr,  config.r, config.mat_type)
        PPR_classify(config.dr, config.sample, sim_mat)
    elif config.task == "task5":
        sim_mat = get_sim_matrix(config.dr, config.r, config.mat_type)
        gestures_list = config.q.split(",")
        relevance_system(config.dr, sim_mat, gestures_list, config.m , config.k, config.c)
    elif config.task == "task6":
        api = GestureApi(config.dr, config.r)
        path = os.path.join(os.getcwd(),"index.html")
        window = webview.create_window('Ppr based object Retreival',
                                       path,
                                       js_api=api,
                                       confirm_close=False,
                                       frameless=False,
                                       min_size=(400, 200))
        webview.start(window)
    else:
        print("Not a valid task")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dr', type=str, default="Phase2_3class_sample_data/3_class_gesture_data/", help='Directory Containing the gesture components')
    parser.add_argument('--task', type=str, default="task0", help='Choose Tasks: task0, task1, task2, task5, task6')
    parser.add_argument('--w', type=int, default = 3, help='Gesture word quantization: Window size')
    parser.add_argument('--r', type=int, default = 3, help='Gesture word quantization: Resolution')
    parser.add_argument('--s', type=int, default = 3, help='Gesture word quantization: Shift length')
    parser.add_argument('--k', type=int, default = 6, help='Transition graph node edges')
    parser.add_argument('--m', type=int, default = 10, help='No of objects to be retrieved')
    parser.add_argument('--c', type=float, default = 0.05, help='PPR constant')
    parser.add_argument('--q', type=str, default = "1", help='Query')
    parser.add_argument('--vect_type', type=str, default = "tfidf", help='Vector Type')
    parser.add_argument('--mat_type', type=str, default = "dot", help='Gesture similarity matrix type')
    parser.add_argument('--sample', type=str, default = "sample_training_labels.xlsx", help='Sample file for training')

    config = parser.parse_args()
    print(config)
    main(config)

#
# Example usage
#
# python3 mwd_pr3.py --help
# python3 mwd_pr3.py --dr Phase2_3class_sample_data/3_class_gesture_data/ --task task0 --w 3 --r 3 --s 3 --vect_type tf --mat_type ed
# python3 mwd_pr3.py --dr Phase2_3class_sample_data/3_class_gesture_data/ --task task1 --m 10 --k 6 --c 0.05 --q "1,2,3"
# python3 mwd_pr3.py --dr Phase2_3class_sample_data/3_class_gesture_data/ --task task2 --sample "samples1.xlsx"
# python3 mwd_pr3.py --dr Phase2_3class_sample_data/3_class_gesture_data/ --task task5 --m 10 --k 6 --c 0.05 --q "1,2,3"
# python3 mwd_pr3.py --dr Phase2_3class_sample_data/3_class_gesture_data/ --task task6
