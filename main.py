import os
import sys
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import collections
import imutils

def remove_punctuation(word):
    #this function is used to remove punctuations
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ""
    for char in word:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct

def read_glove_vecs(glove_file):
    #read pretrained word embeddings, here we form a dictionary of words and their 50 dimensional embeddings
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}

        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

    return words, word_to_vec_map

def compare_single_words(word1, word2, word_to_vec_map):
    #compare single words using cosine similarity
    w1, w2 = word_to_vec_map[word1], word_to_vec_map[word2]
    w1 = np.reshape(w1, (1, -1))
    w2 = np.reshape(w2, (1, -1))
    res = cosine_similarity(w1,w2)
    res = round(res.item(), 4)
    
    return res

def compare_words(input_words, detected_words, word_to_vec_map):
    #compare word embeddings using cosine similarity score. The function returns the average cosine similarity for all comparisons
    total_avg_score = []
    for word1 in input_words:
        avg_scor = []
        for word2 in detected_words:
            avg_scor.append(compare_single_words(word1, word2, word_to_vec_map))
        avg_scor = sum(avg_scor)/len(avg_scor)
    total_avg_score.append(avg_scor)

    return sum(total_avg_score)/len(total_avg_score)

def refine_detections(dets):
   #count frequency of detected objects for multiple detections we simply return a plural for instance persons for 2 or more person
    counter=collections.Counter(dets)
    
    out = []
    for item in counter.items():
        if item[1] > 1:
            out.append(item[0]+'s')
        else:
            out.append(item[0])
            
    return out
    
def object_detection(image, net, THRESHOLD, CLASSES):
    
    #perform a forward pass through the pretrained object detector
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    print("[-] Performing object detections...")
    net.setInput(blob)
    detections = net.forward()

    detected_objects = []
    # iterate detections
    for i in np.arange(0, detections.shape[2]):
        # extract the probability associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        if confidence > THRESHOLD:
            # extract the index of the class label from the `detections`,
            # then compute get the class label from the indes
            
            idx = int(detections[0, 0, i, 1])
            detected_objects.append(CLASSES[idx])

    #get frequency of labels and return plural
    if not detected_objects:
        print('[-][-] No object detected ..exiting !')
    else:
        print('[-][-] Detected objects {}'.format(detected_objects))
        detected_objects = refine_detections(detected_objects)

    return detected_objects

def activity_recognition(image, net,CLASSES):
    frames = []
    #resize the image while maintaining its aspect ratio
    image = imutils.resize(image, width=400)
    frames.append(image)
    #perform a forward pass through the pretrained network and get labels
    blob = cv2.dnn.blobFromImages(frames, 1.0,(112, 112), (114.7748, 107.7354, 99.4750),
		swapRB=True, crop=True)

    blob = np.transpose(blob, (1, 0, 2, 3))
    blob = np.expand_dims(blob, axis=0)

    net.setInput(blob)
    outputs = net.forward()
    activity_label = CLASSES[np.argmax(outputs)]
    print('[-][-] Detected activity {}'.format(activity_label))
    activity_label = activity_label.split(' ')
    
    return activity_label


def main(args):
    
    #for object detection, we load a list of 20 objects and a pretrained neural network
    O_CLASSES = open('object_classes.txt').read().strip().split("\n")
    o_net = cv2.dnn.readNetFromCaffe('object_det_model.prototxt.txt', 'object_det_model.caffemodel')
    THRESHOLD = 0.3

    #for activity recognition load a list of activity classes and a pretrained network
    A_CLASSES = open('action_classes.txt').read().strip().split("\n")
    a_net = cv2.dnn.readNet('activity_model.onnx')

    #load pretrained (GloVe) word embedddings downloaded from https://nlp.stanford.edu/projects/glove/
    words, word_to_vec_map = read_glove_vecs('./glove.6B.50d.txt')

    #get user input (sentence, phrase or word)
    input_words =  args.sentence
    #remove punctuations from words
    input_words = [remove_punctuation(x) for x in input_words]
    
    #load the user supplied image
    image = cv2.imread(args.image_path)

    #check the approach chosen by the user (i.e. 1 or any other number)
    if args.approach == 1:
        detected_words = object_detection(image, o_net, THRESHOLD,O_CLASSES)
        #this uses object detection to find objects, the words can then be compared to user input
        if not detected_words:
            #if non of the 19 object classes is detected, the code exits
            exit(0)
    else:
        detected_words = activity_recognition(image, a_net, A_CLASSES)

    #compute similarity score between detected labels and user input
    similarity_score = compare_words(input_words, detected_words, word_to_vec_map)
    print('[-] Similarity score {}'.format(similarity_score)) 

def parse_args():
    description = \
    '''
    This script can be used to compare text to images.
    Text entered via the command  line is converted into word embedding, then one of two options is used to describe the image.
    
        1) Image is described using objects detected in the image
        
        2) Option two describes and image using activity detection
        
    Usage:
    python3 main.py 
        -i /full/image/path.jpg -s user typed sentence, phrase phrase or word -m 1

    
    '''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--image_path', action='store', help='absolute path to the image', required=True)
    parser.add_argument('-s', '--sentence', action='store', nargs='+', help='sentence to compare', required=True)
    parser.add_argument('-m', '--approach', action='store', type=int, default=1, help='Approach can be 1 or 2', required=False)
        
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)


