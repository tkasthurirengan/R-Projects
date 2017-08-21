#In this example, we are going to look on Dimensionality Reduction. 
#This technique extracts the independent variables which explains most variance in the dataset. 
#For this demonstration I am going to implement Dimensionality reduction with "Principle Component Analysis"

#The data used is the Wine dataset from the UCI Machine Learning Repository and can be found on the following link
#https://archive.ics.uci.edu/ml/datasets/wine

library(caTools)
library(caret)
library(e1071)
setwd("F:/Datasets")
dataset=read.csv("Wine.csv",header = T)
head(dataset)
#When we look at the dataset, we can see that the dependent / Target variable has been clustered based on certain criteria. Hence the main goal is going to use the PCA and SVM to predict the customers which comes under each clusters

# This can also be used as an recommender system, as we know which type of customer will like which sort of wine. 
#We start by splitting the data into test and train. 


set.seed(1234)
split = sample.split(dataset$Customer_Segment, SplitRatio = 0.8)
training = subset(dataset, split == TRUE)
test = subset(dataset, split == FALSE)

#Now, lets do feature scaling to make sure that our data is standardized. 
#In this case, except our target variable every other variables needs to be scaled.

training[-14]=scale(training[-14])
test[-14]=scale(test[-14])

#Lets apply PCA on the scaled dataset. 
#The preProcess function is the place where all the tricks happens. Lets have a look at some important parameters which makes this change
# we need to specify method and we have lot of options like "BoxCox","YeoJohnson", "expoTrans", "center", "scale", "range", "knnImpute", "bagImpute", "medianImpute", "pca", "ica", "spatialSign"
#For this example, we are going to use the "PCA" method
# The next thing is going to Thresh and PCAComp paramerts. If you know the number of components, then there is no need to include thresh as a parameter, as the PCAComp overrides the thresh parameter
# Speaking of thresh, thresh will provide you the features with X% of variance that is required. Example: If we want to reduce the dimensionality which explains at least 70% of variance, then we input thresh=0.7
pca = preProcess(x = training[-14], method = 'pca', pcaComp = 2)

#We now use this PCA object and apply onto our training and test set and finally we re-arrange columns. 
training = predict(pca, training)
training= training[c(2, 3, 1)]
test = predict(pca, test)
test = test[c(2, 3, 1)]

#Will proceed with the classifcation model and I am going to use SVM Classification
library(e1071)
classifier = svm(formula = Customer_Segment ~ .,
                 data = training,
                 type = 'C-classification',
                 kernel = 'linear')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test[-3])

# Making the Confusion Matrix
cm = table(test[, 3], y_pred)
confusionMatrix(cm)

#As we can see from the confusion matrix, we were able to classify the target variable with 94% accuracy.

