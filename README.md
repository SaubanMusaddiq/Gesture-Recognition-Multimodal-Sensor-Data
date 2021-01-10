# Gesture-Recognition-Multimodal-Sensor-Data

Data is taken from http://chalearnlap.cvc.uab.es/dataset/12/description/.

The data consists of Multivariate gesture readings. Sample of the data is included in the repo.
The data consists of 3D spatial coordinates and Rotational orientation of 20 different sensors attached to a human performing gestures.

# Tasks performed:
## Phase1

Task 0: Preprocessing - Normalize, Quantize, Aggregate, and Vectorize(tf, tfidf) the data.
python3 gr_p1.py --dr data/ --task task0 --w 3 --r 3 --s 3

Task 1: Apply dimensionality reduction on the Vectors. NMF is used here.
python3 gr_p1.py --dr data/ --task task1 --k 50

Task 2: Gesture Similarity based on edit distance, top-k latent semantics and other metrics.
python3 gr_p1.py --dr data/ --task task2 --gesture_id 1 --user_option 6 --r 3

Task 3: Latent Gesture Discovery - Reports the top principle components using the gesture-gesture similarity matrix.
python3 gr_p1.py --dr data/ --task task3 --user_option 4 --p 3

Task 4: Clustering using Spectral clustering (task4a) and K-means clustering(task4b).
python3 gr_p1.py --dr data/ --task task4a
python3 gr_p1.py --dr data/ --task task4b --user_option 4 --p 3


python3 gr_p1.py --help

optional arguments:
-h, --help                     show this help message and exit
--dr DR                        Directory Containing the gesture components
--task TASK                    Choose Tasks: task0, task1, task2, task3, task4a,
                               task4c
--w W                          Gesture word quantization: Window size
--r R                          Gesture word quantization: Resolution
--s S                          Gesture word quantization: Shift length
--vector_type VECTOR_TYPE      Vector type tf/Tfidf
--user_option USER_OPTION      User Options 1,4,6
--user_option2 USER_OPTION2    euc/dot
--k K                          k
--p P                          p
--gesture_id GESTURE_ID        Gesture file Name wihout extension. Eg: 1,2,3


## Phase2

Task 0: Preprocessing - Vectorization, gesture-gesture similarity graph creation.
python3 gr_p2.py --dr data/ --task task0 --w 3 --r 3 --s 3

Task 1: Representative Gesture Identification using Personalised page rank Algorithm.
python3 gr_p1.py --dr data/ --task task1 --k 50

Task 2: Gesture classification using PPR algorithm.
python3 gr_p2.py --dr data/ --task task2 --sample "sample_training_labels.xlsx"

Task 5: Classifier with relevance feedback.
python3 gr_p2.py --dr data/ --task task5 --m 2 --k 2 --c 0.05 --q "1,249"

Task 6: Task5 with interactive UI.
python3 gr_p2.py --dr data/ --task task6


python3 gr_p2.py --help

optional arguments:
  -h, --help                show this help message and exit
  --dr DR                   Directory Containing the gesture components
  --task TASK               Choose Tasks: task0, task1, task2, task5, task6
  --w W                     Gesture word quantization: Window size
  --r R                     Gesture word quantization: Resolution
  --s S                     Gesture word quantization: Shift length
  --k K                     Transition graph node edges
  --m M                     No of objects to be retrieved
  --c C                     PPR constant
  --q Q                     Query
  --vect_type VECT_TYPE     Vector Type
  --mat_type MAT_TYPE       Gesture similarity matrix type
  --sample SAMPLE           Sample file for training
