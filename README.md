# Gesture-Recognition-Multimodal-Sensor-Data

Data is taken from http://chalearnlap.cvc.uab.es/dataset/12/description/.

The data consists of Multivariate gesture readings. Sample of the data is included in the repo.</br>
The data consists of 3D spatial coordinates and Rotational orientation of 20 different sensors attached to a human performing gestures.</br>

# Tasks performed:
## Phase1

Task 0: Preprocessing - Normalize, Quantize, Aggregate, and Vectorize(tf, tfidf) the data.</br>
python3 gr_p1.py --dr data/ --task task0 --w 3 --r 3 --s 3

Task 1: Apply dimensionality reduction on the Vectors. NMF is used here.</br>
python3 gr_p1.py --dr data/ --task task1 --k 50

Task 2: Gesture Similarity based on edit distance, top-k latent semantics and other metrics.</br>
python3 gr_p1.py --dr data/ --task task2 --gesture_id 1 --user_option 6 --r 3

Task 3: Latent Gesture Discovery - Reports the top principle components using the gesture-gesture similarity matrix.</br>
python3 gr_p1.py --dr data/ --task task3 --user_option 4 --p 3

Task 4: Clustering using Spectral clustering (task4a) and K-means clustering(task4b).</br>
python3 gr_p1.py --dr data/ --task task4a</br>
python3 gr_p1.py --dr data/ --task task4b --user_option 4 --p 3


python3 gr_p1.py --help

**Optional arguments**:

<table>
<tr><td>-h, --help                 </td><td> Show this help message and exit</td></tr>
<tr><td>--dr DR                    </td><td> Directory Containing the gesture components</td></tr>
<tr><td>--task TASK                </td><td> Choose Tasks: task0, task1, task2, task3, task4a, task4c</td></tr>
<tr><td>--w W                      </td><td> Gesture word quantization: Window size</td></tr>
<tr><td>--r R                      </td><td> Gesture word quantization: Resolution</td></tr>
<tr><td>--s S                      </td><td> Gesture word quantization: Shift length</td></tr>
<tr><td>--vector_type VECTOR_TYPE  </td><td> Vector type tf/Tfidf</td></tr>
<tr><td>--user_option USER_OPTION  </td><td> User Options 1,4,6</td></tr>
<tr><td>--user_option2 USER_OPTION2</td><td> euc/dot</td></tr>
<tr><td>--k K                      </td><td> k</td></tr>
<tr><td>--p P                      </td><td> p</td></tr>
<tr><td>--gesture_id GESTURE_ID    </td><td> Gesture file Name wihout extension. Eg: 1,2,3</td></tr>
</table>

## Phase2

Task 0: Preprocessing - Vectorization, gesture-gesture similarity graph creation.</br>
python3 gr_p2.py --dr data/ --task task0 --w 3 --r 3 --s 3

Task 1: Representative Gesture Identification using Personalised page rank Algorithm.</br>
python3 gr_p1.py --dr data/ --task task1 --k 50

Task 2: Gesture classification using PPR algorithm.</br>
python3 gr_p2.py --dr data/ --task task2 --sample "sample_training_labels.xlsx"

Task 5: Classifier with relevance feedback.</br>
python3 gr_p2.py --dr data/ --task task5 --m 2 --k 2 --c 0.05 --q "1,249"

Task 6: Task5 with interactive UI.</br>
python3 gr_p2.py --dr data/ --task task6


python3 gr_p2.py --help

**Optional arguments**:
<table>
<tr><td>  -h, --help           </td><td>     Show this help message and exit</td></tr>
<tr><td>  --dr DR              </td><td>     Directory Containing the gesture components</td></tr>
<tr><td>  --task TASK          </td><td>     Choose Tasks: task0, task1, task2, task5, task6</td></tr>
<tr><td>  --w W                </td><td>     Gesture word quantization: Window size</td></tr>
<tr><td>  --r R                </td><td>     Gesture word quantization: Resolution</td></tr>
<tr><td>  --s S                </td><td>     Gesture word quantization: Shift length</td></tr>
<tr><td>  --k K                </td><td>     Transition graph node edges</td></tr>
<tr><td>  --m M                </td><td>     No of objects to be retrieved</td></tr>
<tr><td>  --c C                </td><td>     PPR constant</td></tr>
<tr><td>  --q Q                </td><td>     Query</td></tr>
<tr><td>  --vect_type VECT_TYPE</td><td>     Vector Type</td></tr>
<tr><td>  --mat_type MAT_TYPE  </td><td>     Gesture similarity matrix type</td></tr>
<tr><td>  --sample SAMPLE      </td><td>     Sample file for training</td></tr>
</table>

