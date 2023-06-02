# Facial-Expression-Detection-using-CNN-Real-Time-Recognition-with-Webcam
The project aim is to develop a facial expression detection system using CNNs that can accurately recognize and classify facial expressions across a range of emotions. 

## Overview

Facial expression detection has been an active area of research for decades due to its wide range of applications in various fields such as psychology, human-computer interaction, and computer vision. The proposed approach involves processing the input images by scaling and one-hot encoding. It also involves resampling to balance the dataset as the dataset contains the minority class. The pre-processed images are then fed into a CNN model that consist of several convolution layers, batch normalization, activation function, max-pooling layers and fully connected layers. The model is trained using a dataset of facial expression images labelled with the corresponding emotion. The proposed CNN model for facial expression detection aims to accurately classify facial expressions from input images. Here is the complete architecture for proposed CNN model:

1. Input Layer: The input layer of the model receives grayscale images representing faces as input and the dimensions of the input images is given as 48*48. 
2. Convolution Layers: The convolution layers form the backbone of the model and perform feature extraction from the input images. Typically, we define 5 convolution layers with the filter size of 3*3, stride as 1 and padding is sometime “same” and sometime “valid”. These layers are stacked to each other to capture various level of spatial information. Batch Normalization is applied after each layer along with ReLU activation function to introduce non-linearity. 
3. Pooling Layers: Pooling Layers are used to reduce the spatial dimensions of the feature maps and extract the most important features. We have used Max Pooling technique where the maximum value in each local region of the feature map is retained, discarding the rest. Max Pooling helps in reducing computational complexity and providing a form translation invariance to small shifts in the input. 
4. Flattening Layer: After Convolution Layer and Pooling Layer, there is a flattening layer which is responsible for flattened the convolution layers and represent it in a one-dimension.
5. Fully Connected Layers: The fully connected layers are responsible for learning high-level representations of features extracted by the convolution layers. The feature maps from the last pooling layer are flattened into a vector and connected to one or more fully connected layers. Each fully connected layer has a set of learnable weights and biases, which are updated during training process. We have used ReLU activation function and 200 neurons in for our dense layer. And finally, Dropout has been applied after the dense layer to prevent overfitting.
6. Drop Regularization: Dropout Regularization is often employed to prevent overfitting by randomly dropping out a fraction of the neurons during training, forcing the network to rely on different combinations of features.
7. Output Layer:  The output layer of the model performs the classification of facial expressions. For facial expression detection, the output layer typically consists of a SoftMax activation function, which produces the probability scores for each class (e.g., happy, sad, neutral, angry etc.) and the class with the highest probability score is considered as the predicted facial expression.

The two major differences in this architecture compared to other state-of-the-art model are: (a) we have used Batch Normalization after every layer before activation function to normalize the layer, generally it is used after the activation function, and (b) secondly, we have used both kind of padding i.e., valid and same whereas mostly “same” padding has been used in general scenario. 

## Result Analysis:

The proposed approach was evaluated on FER2013 dataset which is publicly available on Kaggle. The results show that the proposed approach outperforms existing state-of-the-art methods in terms of accuracy, precision, recall, and F1-score. The proposed approach achieved an accuracy of 81.39% on FER2013 dataset. The result of this thesis demonstrates the effectiveness of using CNNs for facial expression detection. The proposed approach can be used in various applications such as emotion recognition, human-computer interaction and virtual reality.

## Final Result:

We can use Opencv library to capture the image in a real-time and predict the expression. After running the final piece of code i.e., webcam_test, we will get the below result:


https://github.com/shivamsingh96/Facial-Expression-Detection-using-CNN-Real-Time-Recognition-with-Webcam/assets/123630632/f94ddf6a-a2ff-464e-8b42-e6bde0e7b734

