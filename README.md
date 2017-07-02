# Convolutional-Autoencoder
Implementation of Convolutional Autoencoder which consists of a convolutional encoder and a decoder.
The encoder consists of 3 convolutional layers and 2 fully connected layers in the end. 
On the other hand, the decoder consists of 2 fully connected layers and 3 transposed convolutional layers.
Adam Optimizer is used for learning.

The whole code is implemented in Tensorflow slim.

P.S - This is an exemplary code so it has been implemented for just 1 image. To train on a directory of images, some changes will have to be made in the utils_preprocess code.

