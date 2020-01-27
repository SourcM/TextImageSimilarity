# TextImageSimilarity

##Description
The script *main.py* is used to compare an input text to the description of an input image. 
- GloVe pretrained word embedding vectors are used to represent words as vectors
- using one of two methods, the input image is described in words. 
-Subsequently the embeddings of the description are compared to the input text
  - we iteratively compare each input word to the out text, then all similarity scores are averaged in order to given a single number representing the overall average similarity score

###Image representation as text
- The first approach used to represent image as text is via the use of a pretrained (Mobilenet) object detection algorithm to classify the objects in the image into one of 20 classes listed in *object_classes.txt*
- The second approach uses a pretrained activity recognition algorithm to classify an image into one of 400 activities listed in *activity_classes.txt*
    
##Usage
The algorithm can be tested via running the command (written below) in the terminal
python3 main.py -i full_path_to_image -s words,sentence or phrase -m approach(this can be 1 or 2) example

```
python3 -i image.jpg -s man is on bike -m 1

```
##Requirements
pip install numpy
pip install sklearn
pip install imutils
pip install opencv-python==4.1.2

##Additional Files
Download GloVe pretrained word vectors and pretrained activity recognition network [here](https://pages.github.com/)
Insert the downloaded files into the root directory containing the _main.py_ script, as well as other resources.

##Limitation
Outcome of the pipeline presented in this project relies on one of the classes to which the pretrained models is able to categorise objects/activities. The result is also affected by image resolution and size. 
