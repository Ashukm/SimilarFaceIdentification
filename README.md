# SimilarFaceIdentification
The document describes an approach to find similar images for an given input image from the publicly available CelebA dataset.
1 Problem Statement
Given an input face image , find 5 similar images to the input image from the
CelebA database.

2 Database

The CelebA database is a celebrity facial images database that provides 40 attributes which describe different facial features of each image. Each attribute
takes a value 1 or -1 depending on the image content e.g wearing hat -1 = No ,
1 = Yes. I used only around 5000 images from the CelebA database to validate
the approach. Around 3000 were used for training. 1000 for validation and 1000
for testing.

3 Approach

The problem is solved in two parts. The first part tries to predict the 40 attributes
of the given input image using a deep model. In the second part, given the
predicted attributes, the 5 closest neighbors to the image are identified using
supervised KNN.
3.1 Attribute Prediction
The approach uses pre-trained ResNet152 model. The ResNet152 is trained using
a subset of CelebA images with their corresponding attributes. The inputs were
the images and the targets were the 40 attributes. The final layer of the model
is replaced with multiple binary classification heads. One binary classification
head for every attribute. Losses for all heads are summed up together during
backward propagation. After the training, given an input image the the model
outputs the predicted 40 attributes.

3.2 Identifying Neighbors
A KNN model is fitted which takes the 40 facial attributes as input features and
the image name or index as output. All 5000 images are used to fit the KNN.
Once the deep model predicts the attributes, the predicted attributes as passed
as input feature to a KNN. Its kneighbors function is then used to get the closest
5 images to the given features.
