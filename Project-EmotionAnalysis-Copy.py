#Required Libraries
#from _typeshed import NoneType
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
import pandas as pd
import time
import argparse
import imutils
import sys
import networkx as nx
from collections import deque
from random import randint
#from Image_captioning import graph as ICgraph
from Image_captioning import caption

# Librairy to draw the graph
import re
import pandas as pd
import bs4
import requests
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

from spacy.matcher import Matcher 
from spacy.tokens import Span 

import networkx as nx

import matplotlib.pyplot as plt
from tqdm import tqdm

pd.set_option('display.max_colwidth', 200)
#%matplotlib inline

import get_relation
import get_entities



#-------------------------------------------------------------------------------------------

#Web Application View

#Defining the Layout

st.title('EMOTION-ANALYSIS')
l1,l2,l3,l4 = st.beta_columns([2,2,3,2])
option = l1.selectbox('Select the View',
                      ('Outdoor View', 'Indoor View', 'Sample Video'))
stream = l1.button("STREAM NOW")


#Capturing the Video stream

if option == 'Outdoor View':
    try:
        cam = cv2.VideoCapture("http://81.149.56.38:8081/mjpg/video.mjpg")
        cam.set(cv2.CAP_PROP_FPS, 1)
    except Exception as e:
        print("Error :", e)
        
else:
    if option == 'Indoor View':
        try:
            cam = cv2.VideoCapture("http://81.83.10.9:8001/mjpg/video.mjpg")
            cam.set(cv2.CAP_PROP_FPS, 1)
        except Exception as e:
            print("Error :", e)
    else:
        cam = cv2.VideoCapture("cream.mp4")
        cam.set(cv2.CAP_PROP_FPS, 1)

#stop = st.button("STOP NOW")
#if stop:
#    exit(0)
my_placeholder1 = l1.empty()
my_placeholder2 = l2.empty()
my_placeholder3 = l3.empty()
my_placeholder4 = l4.empty()
#------------------------------------------------------------------------------------------

#Part -1 : Human Keypoints Detection
#parser = argparse.ArgumentParser(description='Run keypoint detection')
#parser.add_argument("--device", default="cpu", help="Device to inference on")
#parser.add_argument("--image_file", default="Friends.jpg", help="Input image")
#args = parser.parse_args()

protoFile = "pose_deploy_linevec.prototxt"
weightsFile = "pose_iter_440000.caffemodel"
nPoints = 18
# COCO Output Format
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16] ]

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
          [47,48], [49,50], [53,54], [51,52], [55,56],
          [37,38], [45,46]]

colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]


def getKeypoints(probMap, threshold=0.1):

    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)

    mapMask = np.uint8(mapSmooth>threshold)
    keypoints = []

    #find the blobs
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints


# Find valid connections between the different joints of a all persons present
def getValidPairs(output):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid

        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else: # If no keypoints are detected
            print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs



# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
def getPersonwiseKeypoints(valid_pairs, invalid_pairs):
    # the last number in each row is the overall score
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints
#-----------------------------------------------------------------------------------------------------------------------------#

#Part -2 : DeepFace Analysis
font = cv2.FONT_HERSHEY_SIMPLEX
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

#-----------------------------------------------------------------------------------------------------------------------------#

#Part -3 : Action Recognition
#Loading Action Recognition Model

tab=[]
n={}

args = {
    'model': 'resnet-34_kinetics.onnx',
    'classes': 'action_recognition_kinetics.txt'
    }
    
CLASSES = open(args["classes"]).read().strip().split("\n")
SAMPLE_DURATION = 1
SAMPLE_SIZE = 112
frames = deque(maxlen=SAMPLE_DURATION)
#st.write("Loading the Action Recognition Model...")
net1 = cv2.dnn.readNet(args['model'])
#st.write("[INFO] DONE LOADING")

def action_recognition(ret,frame):
    n={}
    #i=0
    #l2.write("Action Function called")
    #Action Recognition
    if not ret:
        l2.write("[INFO] no frame read from stream - exiting")
        #break
    else:
        #l2.write("[INFO] Action Recognising")
        frame = imutils.resize(frame, width=400)
        frames.append(frame)
        #if len(frames) < SAMPLE_DURATION:
            #continue
        blob = cv2.dnn.blobFromImages(frames, 1.0,
                                      (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
                                      swapRB=True, crop=True)
        
        blob = np.transpose(blob, (1, 0, 2, 3))
        blob = np.expand_dims(blob, axis=0)
        net1.setInput(blob)
        outputs = net1.forward()
        idx = np.argsort(outputs[0])[::-1][0]
        #l2.write(idx)
        #To get the class reference value
        n=np.append(n,idx+1,axis=None)
        action =CLASSES[np.argmax(outputs)]
        action_per = (outputs[0][idx]*10)
        #i=i+1
        #l2.write(action)
        return action,action_per





#-----------------------------------------------------------------------------------------------------------------------------#

#Part -4 : Knowledge Graph

def KG_DATA(result,action,action_per,count):
    #Initializing the Node Values
    age = result['age']
    gender = result['gender']
    race = result['dominant_race']
    race_per= round(result['race'][race],2)
    emotion = result['dominant_emotion']
    emotion_per = round(result['emotion'][emotion],2)
    action_per = round(action_per,2)
    
    #Constructing the Knowledge Graph
    head = "HUMAN"
    kg = nx.OrderedGraph()
    kg.add_node(head, size = 5000)
    kg.add_edge(age,head)
    kg.add_edge(gender,head)
    kg.add_edge(race,head)
    kg.add_edge(race_per,race)
    kg.add_edge(emotion,head)
    kg.add_edge(emotion_per,emotion)
    kg.add_edge(head,action)
    kg.add_edge(action_per,action)
    pos = nx.kamada_kawai_layout(kg)
    fig = plt.figure(figsize=(6,5))
    nx.draw(kg, pos=pos,font_size=10,with_labels=True, node_size=800, node_color="aqua", width=2)
    nx.draw_networkx_edge_labels(kg,pos,edge_labels={(age,head):'Age',(gender,head):'Gender',(race,head):'Race',
                                                     (race_per,race):'Probability',(emotion,head):'Emotion',
                                                     (emotion_per,emotion):'Probability',
                                                     (head,action):'Action',(action_per,action):'Probability'},
                                 label_pos=0.5, font_size=8, font_color='red', font_family='sans-serif', font_weight='normal')
    plt.savefig('graph.png')
    my_placeholder2.image('graph.png')
    shape = fig.get_size_inches()
    #my_placeholder2.write(shape)
    #my_placeholder2.pyplot(fig, figsize = (50,50))
    
    cv2.destroyAllWindows()
#-----------------------------------------------------------------------------------------------------------------------------#

#Part -5 : Knowledge Graph Caption

def KG_DATA2(descriptionclean):
    entity_pairs = []

    for i in range(len(descriptionclean)):
        entity_pairs.append(get_entities.get_entities(descriptionclean[i]))


    relations = [get_relation.get_relation(descriptionclean[i]) for i in range(len(descriptionclean))]

# extract subject
    source = [i[0] for i in entity_pairs]
     
# extract object
    target = [i[1] for i in entity_pairs]
     
#table of relation in the sentences
    kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})
    print(kg_df)

    
    #Ploting of the graph
    #fig = plt.figure(figsize=(12,12))
    plt.figure(figsize=(12,12))

    # create a directed-graph from a dataframe
    G = nx.from_pandas_edgelist(kg_df, "source", "target", 
                              edge_attr=True, create_using=nx.MultiDiGraph())

    pos = nx.spring_layout(G)
    nx.draw(G, pos, edge_color='black', width=1, linewidths=3,
            node_size=300, node_color='skyblue', alpha=1,node_shape="s",
            labels={node: node for node in G.nodes()})
    edges = nx.get_edge_attributes(G, 'edge')
    edge_labels = {i[0:2]:'{}'.format(i[2]['edge']) for i in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels,label_pos=0.5, font_size=12)
    plt.savefig('langmodel22.png')
    my_placeholder4.image('langmodel22.png')
    shape = fig.get_size_inches()
    #my_placeholder2.write(shape)
    #my_placeholder2.pyplot(fig, figsize = (50,50))
    
    cv2.destroyAllWindows()
#-----------------------------------------------------------------------------------------------------------------------------#

#Main Program

description=[]
count=1
device="gpu"
if stream:

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    if device == "cpu":
        net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        print("Using CPU device")
    elif device == "gpu":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using GPU device")


    ##Captioning class object
    cap = caption.ImageCaptioning(model='Image_captioning//New_Model.h5',tokenizer='Image_captioning//New_Tok.pkl')
    caption_textList = 'Ligne1\nLigne2\nLigne3\nLigne4\nLigne5\nLigne6\nLigne7\nLigne8\nLigne9\nLigne10'
    st.text_area("Description model", value=caption_textList, height=275, max_chars=0, key=10)

    while True:
        ret,frame=cam.read()
        image1=frame
        #if(type(frame) == NoneType):
        #continue
        frameWidth = image1.shape[1]
        frameHeight = image1.shape[0]
        t = time.time()
        # Fix the input Height and get the width according to the Aspect Ratio
        inHeight = 368
        inWidth = int((inHeight/frameHeight)*frameWidth)
        inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
                                        (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()
        print("Time Taken in forward pass = {}".format(time.time() - t))
        detected_keypoints = []
        keypoints_list = np.zeros((0,3))
        keypoint_id = 0
        threshold = 0.1
        
        ##Captioning
        descriptionclean=[]
        new_caption = cap.run(frame)
        print(new_caption)
        caption_textList = new_caption
        description.append(new_caption)
        for i in range(len(description)):
            descriptionclean.append(description[len(description)-i][9:(len(description[len(description)-i])-7)])
            if len(description)>10:
                descriptionclean.append(description[len(description)-10][9:(len(description[len(description)-10])-7)])
                descriptionclean.append(description[len(description)-9][9:(len(description[len(description)-9])-7)])
                descriptionclean.append(description[len(description)-8][9:(len(description[len(description)-8])-7)])
                descriptionclean.append(description[len(description)-7][9:(len(description[len(description)-7])-7)])
                descriptionclean.append(description[len(description)-6][9:(len(description[len(description)-6])-7)])
                descriptionclean.append(description[len(description)-5][9:(len(description[len(description)-5])-7)])
                descriptionclean.append(description[len(description)-4][9:(len(description[len(description)-4])-7)])
                descriptionclean.append(description[len(description)-3][9:(len(description[len(description)-3])-7)])
                descriptionclean.append(description[len(description)-2][9:(len(description[len(description)-2])-7)])
                descriptionclean.append(description[len(description)-1][9:(len(description[len(description)-1])-7)])
            

        for part in range(nPoints):
            probMap = output[0,part,:,:]
            probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
            keypoints = getKeypoints(probMap, threshold)
            print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
            keypoints_with_id = []
            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                keypoint_id += 1
            
            detected_keypoints.append(keypoints_with_id)
            
        frameClone = image1.copy()
        for i in range(nPoints):
            for j in range(len(detected_keypoints[i])):
                cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)
                #cv2.imshow("Keypoints",frameClone)
                
        valid_pairs, invalid_pairs = getValidPairs(output)
        personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs)
        
        for i in range(17):
            for n in range(len(personwiseKeypoints)):
                index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                if -1 in index:
                    continue
                    
                B = np.int32(keypoints_list[index.astype(int), 0])
                A = np.int32(keypoints_list[index.astype(int), 1])
                cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
                    
        #cv2.imshow("Detected Pose" , frameClone)
        shape = frameClone.shape
        width = int(shape[1]*0.6)
        height = int(shape[0]*0.6)
        dimension = (width,height)
        frameClone = cv2.resize(frameClone,dimension,interpolation = cv2.INTER_AREA)
        frameClone = cv2.cvtColor(frameClone,cv2.COLOR_BGR2RGB)
        my_placeholder1.image(frameClone)
                
        try:
            #Deepface Analysis
            result = DeepFace.analyze(frame)
            my_placeholder2.write(result)
            #Action Recognition
            action,action_per = action_recognition(ret,frame)#To store action analysis result
            my_placeholder2.write(action)
            #KnowledgeGraph
            KG_DATA(result,action,action_per,count)
            count=count+1
        except Exception as e:
            my_placeholder2.write("Faces are not clear")
            #my_placeholder2.write(e)

        try:
            #KnowledgeGraph caption
            KG_DATA2(descriptionclean)
            count=count+1
        except Exception as e:
            my_placeholder4.write("Need more description")
       
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows() 



