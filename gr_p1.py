import pandas as pd
import scipy.integrate as integrate
import numpy as np
from itertools import accumulate
import matplotlib.pyplot as plt

from collections import defaultdict
import os
import math
from os import listdir
from sklearn.decomposition import NMF
import argparse




# Util

def get_no_of_sensors(dr):
    with open(os.path.join(dr, [f for f in os.listdir(dr) if f.endswith('.csv')][1]), 'r') as file:
        no_of_sensors = len(file.readlines())
    return no_of_sensors



# # Task 0a

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


def create_gesture_wordfiles_ofComponent(dr, comp_name,  w = 4, s = 3, r = 3):
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
        create_gesture_wordfiles_ofComponent(dr,comp,w,s,r)


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


# Task 1

def get_all_gesture_ids(dr):
    onlyfiles = [f for f in listdir(dr) if os.path.isfile(os.path.join(dr, f))]
    if(len(onlyfiles) == 0):
        print("Task 0 not completed: Create Wrd files")
        return
    gesture_files = [f.split(".")[0] for f in onlyfiles]
    gesture_files.sort()
    return gesture_files

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


def extraxt_latent_topic_word_scores(dr, H, index_word_mp):
    latent_topic_word_scores = {}
    truncated_topic_word_scores ={}
    for lt in range(H.shape[0]):
        word_scores = []
        h = H[lt]
        sorted_index = h.argsort()[::-1]
        for ix in sorted_index:
            if(h[ix] == 0):
                break
            word_scores.append((index_word_mp[ix], h[ix]))
        latent_topic_word_scores["latent_topic_"+str(lt)] = word_scores
        truncated_topic_word_scores["latent_topic_"+str(lt)] = word_scores[:100]
        print("latent_topic_"+str(lt)+" : "+ str(word_scores[:3]))
    with open(os.path.join(dr+"meta_data/", "latent_topic_word_scores.txt"), 'w') as file:
        file.write(str(latent_topic_word_scores))
    with open(os.path.join(dr+"meta_data/", "truncated_topic_word_scores.txt"), 'w') as file:
        file.write(str(latent_topic_word_scores))


def extract_topk_latent_semantics(dr, vector_type = "tf", k =50):
    gesture_files = get_all_gesture_ids(dr+"gesture_words/")
    gesture_index = {ky: v for v, ky in enumerate(gesture_files)}
    with open(os.path.join(dr+"meta_data/", "gesture_index.txt"), 'w') as file:
        file.write(str(gesture_index))
    no_of_gestures = len(gesture_files)
    no_of_sensors =  get_no_of_sensors(dr+"X")

    data_mat, index_word_mp = create_data_mat(dr, vector_type, gesture_files, no_of_gestures, no_of_sensors)
    with open(os.path.join(dr+"meta_data/", "data_mat.npy"), 'wb') as f:
        np.save(f, data_mat)
    nmf = NMF(n_components=k, init='random', random_state=0)
    nmf_W = nmf.fit_transform(data_mat)
    with open(os.path.join(dr+"meta_data/", "data_mat_nmf_W.npy"), 'wb') as f:
        np.save(f, nmf_W)
    nmf_H = nmf.components_
    with open(os.path.join(dr+"meta_data/", "data_mat_nmf_H.npy"), 'wb') as f:
        np.save(f, nmf_H)
    extraxt_latent_topic_word_scores(dr, nmf_H, index_word_mp)




# Task 2

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

def get_most_similar_gestures(dr, gesture_id, method, dist_type, n, r):
    with open(os.path.join(dr+"meta_data/", "gesture_index.txt"), 'r') as file:
        gesture_index_map = eval(file.read())
    index_gesture_map = {v:k for k,v in gesture_index_map.items()}

    if method == "1": # Dot Product
        with open(os.path.join(dr+"meta_data/", "data_mat.npy"), 'rb') as f:
            data_mat = np.load(f)
        gesture_vec = data_mat[gesture_index_map[gesture_id]].reshape(1,data_mat.shape[1])
        dist_arr = np.dot(gesture_vec,data_mat.T)[0]
        top_n_index = dist_arr.argsort()[-n:][::-1]
        top_n_score = dist_arr[top_n_index]
        top_n_gesture = [index_gesture_map[i] for i in top_n_index]

    elif method == "4": # NMF Latent Dimensions
        with open(os.path.join(dr+"meta_data/", "data_mat_nmf_W.npy"), 'rb') as f:
            data_mat = np.load(f)
        gesture_vec = data_mat[gesture_index_map[gesture_id]].reshape(1,data_mat.shape[1])
        if dist_type =="dot": # Dot Product Sim on NMF mat
            dist_arr = np.dot(gesture_vec,data_mat.T)[0]
            top_n_index = dist_arr.argsort()[-n:][::-1]
            top_n_score = dist_arr[top_n_index]
            top_n_gesture = [index_gesture_map[i] for i in top_n_index]
        else:               # Euc distance on NMF mat
            euc_ds = np.linalg.norm(data_mat - gesture_vec, axis=1)
            top_n_index = euc_ds.argsort()[:n]
            top_n_score = euc_ds[top_n_index]
            top_n_gesture = [index_gesture_map[i] for i in top_n_index]

    elif  method == "6": # Edit Distance
        gesture_dists = get_gesture_edit_distances(dr,gesture_id,r)
        gesture_dists = np.array(gesture_dists)
        top_n_index = gesture_dists.argsort()[:n]
        top_n_score = gesture_dists[top_n_index]
        top_n_gesture = [index_gesture_map[i] for i in top_n_index]
    else:
        print("Not a valid user option")
        return
    return list(zip(top_n_gesture,top_n_score))

# Task 3a

def get_gest_mat_edit_distance(dr, gesture_files, r):
    out_mat = []
    for file_id in gesture_files:
        gest_dist = get_gesture_edit_distances(dr, file_id, r)
        out_mat.append(gest_dist)
    gesture_mat = np.array(out_mat)
    return gesture_mat

def create_gesture_gesture_mat(dr, method = "4", r = 3):
    if(method == "1"):
        with open(os.path.join(dr+"meta_data/", "data_mat.npy"), 'rb') as f:
            data_mat = np.load(f)
        gest_mat = np.dot(data_mat,data_mat.T)
        with open(os.path.join(dr+"meta_data/", "ges_ges_mat_DP.npy"), 'wb') as f:
            np.save(f, gest_mat)
        with open(os.path.join(dr+"meta_data/", "ges_ges_mat_DP.csv"), 'w') as f:
            np.savetxt(f, gest_mat, delimiter=",")
    elif(method == "4"):
        with open(os.path.join(dr+"meta_data/", "data_mat_nmf_W.npy"), 'rb') as f:
            in_mat = np.load(f)
        gest_mat = np.dot(in_mat,in_mat.T)
        with open(os.path.join(dr+"meta_data/", "ges_ges_mat_nmf.npy"), 'wb') as f:
            np.save(f, gest_mat)
        with open(os.path.join(dr+"meta_data/", "ges_ges_mat_nmf.csv"), 'w') as f:
            np.savetxt(f, gest_mat, delimiter=",")

    elif(method == "6"):
        gesture_files = get_all_gesture_ids(dr+"gesture_words/")
        in_mat = get_gest_mat_edit_distance(dr, gesture_files, r)
        gest_mat = 1/(1+in_mat)
        with open(os.path.join(dr+"meta_data/", "ges_ges_mat_ed.npy"), 'wb') as f:
            np.save(f, gest_mat)
        with open(os.path.join(dr+"meta_data/", "ges_ges_mat_ed.csv"), 'w') as f:
            np.savetxt(f, gest_mat, delimiter = ",")

    print("Created gesture-gesture similarity matrix and saved in mata data folder")

def principal_components_svd(p, data_mat):
    u, s, vh = np.linalg.svd(data_mat)
    V = vh[:p,:]
    return V

def get_p_principal_components(dr, p, method="4"):
    with open(os.path.join(dr+"meta_data/", "gesture_index.txt"), 'r') as file:
        gesture_index_map = eval(file.read())
    index_gesture_map = {v:k for k,v in gesture_index_map.items()}
    p_dict = {}

    if(method == "1"):
        with open(os.path.join(dr+"meta_data/", "ges_ges_mat_DP.npy"), 'rb') as f:
            data_mat = np.load(f)
    elif(method == "4"):
        with open(os.path.join(dr+"meta_data/", "ges_ges_mat_nmf.npy"), 'rb') as f:
            data_mat = np.load(f)
    elif(method == "6"):
        with open(os.path.join(dr+"meta_data/", "ges_ges_mat_ed.npy"), 'rb') as f:
            data_mat = np.load(f)
    V = principal_components_svd(p, data_mat)
    with open(os.path.join(dr+"meta_data/", "similarity_mat_reduced.npy"), 'wb') as f:
        np.save(f,V)
    ind = np.argsort(V)
    i = 0
    for vh in V:
        val = []
        ind_row = ind[i][::-1]
        for j in range(V.shape[1]):
            n = int(ind_row[j])
            g_val = index_gesture_map[n]
            val.append((g_val , vh[n]))
        p_dict["latent_topic_"+str(i)] = val
        i += 1

    with open(os.path.join(dr+"meta_data/", "svd_latent_gesture_score.txt"), 'w') as f:
        f.write(str(p_dict))

    print("SVD Done on similarity matrix, saved as svd_latent_gesture_score.txt")

#task4a

def get_latent_topic_gesture_group_map(dr):
    with open(os.path.join(dr+"meta_data/", "gesture_index.txt"), 'r') as file:
        gesture_index_map = eval(file.read())
    index_gesture_map = {v:k for k,v in gesture_index_map.items()}
    with open(os.path.join(dr+"meta_data/", "similarity_mat_reduced.npy"), 'rb') as f:
        svd_vh = np.load(f)
    pgroups  = np.argmax(svd_vh, axis=0).tolist()
    latent_topic_gesture_group_map = defaultdict(list)
    for i,g in enumerate(pgroups):
        latent_topic_gesture_group_map[g].append(index_gesture_map[i])

    return latent_topic_gesture_group_map


# Task 4c

def kmeans_cluster(data_mat, k):
    centroids = data_mat[np.random.choice(data_mat.shape[0], k, replace=False), :]
    old_centroid = np.zeros(data_mat.shape[0])
    for lp in range(200):
        nearest_centroid = np.zeros(data_mat.shape[0])
        for i in range(data_mat.shape[0]):
            row = data_mat[i]
            da = np.zeros(centroids.shape[0])
            for j in range(centroids.shape[0]):
                da[j] = np.sqrt(((row-centroids[j])**2).sum())
            nearest_centroid[i] = np.argmin(da)

        if((old_centroid == nearest_centroid).all()):
            print("After {} iterations of Kmeans algorithm".format(lp))
            break
        for j in range(centroids.shape[0]):
            centroids[j] = np.mean(data_mat[nearest_centroid == j],axis = 0)

        old_centroid = np.copy(nearest_centroid)

    return old_centroid

def cluster_gestures(dr, method="1", k = 3):
    if(method == "1"):
        print("Using Dot Product Similarity Matrix")
        with open(os.path.join(dr+"meta_data/", "ges_ges_mat_DP.npy"), 'rb') as f:
            data_mat = np.load(f)
    elif(method == "4"):
        print("Using Dot Product Similarity Matrix of NMF Reduced Data")
        with open(os.path.join(dr+"meta_data/", "ges_ges_mat_nmf.npy"), 'rb') as f:
            data_mat = np.load(f)
    elif(method == "6"):
        print("Using Dot Product Similarity Matrix obtained from edit distance")
        with open(os.path.join(dr+"meta_data/", "ges_ges_mat_ed.npy"), 'rb') as f:
            data_mat = np.load(f)

    centres = kmeans_cluster(data_mat, k)

    with open(os.path.join(dr+"meta_data/", "gesture_index.txt"), 'r') as file:
        gesture_index_map = eval(file.read())
    index_gesture_map = {v:k for k,v in gesture_index_map.items()}

    k_clusters = defaultdict(list)
    for i in range(data_mat.shape[0]):
        k_clusters[int(centres[i])].append(index_gesture_map[i])

    print("Clusters",k_clusters)
    with open(os.path.join(dr+"meta_data/", "ges_cluster.txt"), 'w') as f:
        f.write(str(k_clusters))



def main(cofig):
    if not os.path.exists(config.dr+"vectors"):
        os.makedirs(config.dr+"vectors")
    if not os.path.exists(config.dr+"meta_data"):
        os.makedirs(config.dr+"meta_data")

    if config.task == "task0":
        create_gesture_wordfiles(config.dr ,config.w, config.s, config.r)
        no_of_sensors =  get_no_of_sensors(config.dr+"X")
        create_gesture_vectors(config.dr, no_of_sensors = 20, comps = ["X","Y","Z","W"])
    elif config.task == "task1":
        extract_topk_latent_semantics(config.dr, config.vector_type, config.k)
    elif config.task == "task2":
        l = get_most_similar_gestures(config.dr, config.gesture_id, config.user_option, config.user_option2, 10, config.r)
        print("list of most similar gestures")
        print(l)
    elif config.task == "task3":
        create_gesture_gesture_mat(config.dr, config.user_option, config.r)
        get_p_principal_components(config.dr, config.p, config.user_option)
    elif config.task == "task4a":
        g = get_latent_topic_gesture_group_map(config.dr)
        print("Gestures grouped to latent topics discovered in task3")
        print(g)
    elif config.task == "task4b":
        cluster_gestures(config.dr, config.user_option, config.p)
    else:
        print("Not a valid task")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dr', type=str, default="Phase2_3class_sample_data/3_class_gesture_data/", help='Directory Containing the gesture components')
    parser.add_argument('--task', type=str, default="task0", help='Choose Tasks: task0, task1, task2, task3, task4a, task4c')
    parser.add_argument('--w', type=int, default = 3, help='Gesture word quantization: Window size')
    parser.add_argument('--r', type=int, default = 3, help='Gesture word quantization: Resolution')
    parser.add_argument('--s', type=int, default = 3, help='Gesture word quantization: Shift length')
    parser.add_argument('--vector_type', type=str, default = "tf", help='Vector type tf/Tfidf')
    parser.add_argument('--user_option', type=str, default = "4", help='User Options 1,4,6')
    parser.add_argument('--user_option2', type=str, default = "dot", help='euc/dot')
    parser.add_argument('--k', type=int, default = 20, help='k')
    parser.add_argument('--p', type=int, default = 3, help='p')
    parser.add_argument('--gesture_id', type=str, default = "1", help='Gesture file Name wihout extension. Eg: 1,2,3')
    config = parser.parse_args()
    print(config)
    main(config)

#
# Example usage
#
# python3 mwd_pr2.py --help
# python3 mwd_pr2.py
# python3 mwd_pr2.py --dr Phase2_3class_sample_data/3_class_gesture_data/ --task task0 --w 3 --r 3 --s 3
# python3 mwd_pr2.py --dr Phase2_3class_sample_data/3_class_gesture_data/ --task task1 --k 50
# python3 mwd_pr2.py --dr Phase2_3class_sample_data/3_class_gesture_data/ --task task2 --gesture_id 1 --user_option 6 --r 3
# python3 mwd_pr2.py --dr Phase2_3class_sample_data/3_class_gesture_data/ --task task3  --user_option 4 --p 10
# python3 mwd_pr2.py --dr Phase2_3class_sample_data/3_class_gesture_data/ --task task4a
# python3 mwd_pr2.py --dr Phase2_3class_sample_data/3_class_gesture_data/ --task task4b  --user_option 4 --p 3
