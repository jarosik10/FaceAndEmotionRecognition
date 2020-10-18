# FaceAndEmotionRecognition
## Description
An application that recognizes the identity of the user and emotions based on facial images obtained in real time via webcam. The identity recognition system was created using the FaceNet model, which was incorporated to create the feature vector representing the face, and the SVM algorithm, which was used as the classifier. To create an emotion recognition system, a neural network was taught using a machine learning method called Transfer Learning. 
## Results
![Face recognition example](/images/face_recognition.png)
![Emotion recognition example](/images/emotion_recognition.png)

Accuracy of models:
- Face recognition system: **98,0%**
- Emotion recognition system: **78,1%**

## Technologies:
- Tensorflow 2.0
- Keras 2.3.1
- Sckit-learn
- OpenCV

## Datasets:
- Pins Face Recognition (face recognition dataset)
- FER2013 (emotion recognition dataset)
- CK+ (emotion recognition dataset)
