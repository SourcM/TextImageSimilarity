# TextImageSimilarity

## Description
The script *main.py* is used to compare an input text to the description of an input image. 
- GloVe pretrained word embedding vectors are used to represent words as vectors
- Using one of two methods, the input image is described in words 
- Subsequently, the embeddings of the description are compared to the input text
  - we iteratively compare each input word to the word representation of the image, then all similarity scores are averaged in order to get a single number representing the overall average similarity score
- It is worth noting that punctuation marks are not considered

### Image representation as text
- The first approach used to represent image as text is via the use of a pretrained (Mobilenet) object detection algorithm to classify the objects in the image into one of 20 classes listed in *object_classes.txt*
   - An objected detected multiple times is represented as the plural form of the detected label
- The second approach uses a pretrained activity recognition algorithm to classify an image into one of 400 activities listed in *activity_classes.txt*
    
## Usage
The algorithm can be tested by running the command (written below) in the terminal
python3 main.py -i full_path_to_image -s words,sentence or phrase -m approach(this can be 1 or 2) 
for example:

```
python3 -i image.jpg -s man is on bike -m 1

```
## Requirements
- pip install numpy
- pip install sklearn
- pip install imutils
- pip install opencv-python==4.1.2.30

## Additional Resources
Download GloVe pretrained word vectors and pretrained activity recognition network [here](https://drive.google.com/open?id=1thhyWmmg7jANBUcLK8gr2x0B3Rc75ZEF)
Insert the downloaded files into the root directory containing the _main.py_ script.

## Limitation
Outcome of the pipeline presented in this project relies on one of the classes to which the pretrained models is able to categorise objects/activities into. The result is also affected by image resolution and size. 
