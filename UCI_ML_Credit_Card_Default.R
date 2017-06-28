#This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.
#Source: UCI Machine Learning.

library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(caTools)
library(class)
library(caret)
library(ROCR)
library(pROC)
library(factoextra)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


dataset = read.csv("../input/UCI_Credit_Card.csv")

#Lets do a pre processing of data, and before that lets have a look at the structure of the dataset. 
str(dataset)

#From the results we see that there are variables which has to be converted to factors(SEX,MARRIAGE,EDUCATION,PAY_0,PAY_2 TO PAY_6,DEFAULT. 
#Lets Start with dropping the ID, as we wont get any informations from them.
dataset = dataset[-1]
datafactor <-c(2,3,4,6:11,24)
dataset[,datafactor]<-lapply(dataset[,datafactor],factor)

#After the conversion, we have certain variables which are continious. Being continious data is prone is non-standardization. To make the data standardise, we use feature scaling to do the same.
datasetscaled1 <- c(1,5,12:23)
dataset[,datasetscaled1]<-lapply(dataset[,datasetscaled1],scale)

#Now, lets take a final look at how our dataset is structured. 
str(dataset)

#we can see from the output that our data is cleaned and processed the way we need to do. 
#Lets Visualize the given dataset to get the feel of the dataset. 
#Kindly note, we are not drawing any conclusion from the graphs below. It is just an overview of how the data is spread across the dataset. 
#The following plot compares the Gender with the defaulters. This is to get an idea of which gender is defaulting most. 

GDS<-ggplot(dataset,aes(dataset$default.payment.next.month,fill=SEX))
GDS+geom_bar(position="dodge")+coord_flip()+ggtitle("Comparison of Sex Vs Defaulters")+xlab("Default - Yes / No")+ylab("Count of Persons")

#Education plays a major role in identifying which class of people are defaulting the most. With the graph below we can get an overview. 

DEdu<-ggplot(dataset,aes(dataset$default.payment.next.month,fill=EDUCATION))
DEdu+geom_bar(position="stack")+ggtitle("Comparison of Education Vs Defaulters")+xlab("Default - Yes / No")+ylab("Count of Persons")

#This is another level of visualization to check about the defaulters. 

Mar<-ggplot(dataset,aes(dataset$default.payment.next.month,fill=MARRIAGE))
Mar+geom_bar(position="dodge")+ggtitle("Comparison of Marriage Vs Defaulters")+xlab("Default - Yes / No")+ylab("Count of Persons")


#Lets roll on to the problem. It is always good to split the data into test and train. In this case, i am taking 70% into training. 

set.seed(12345)
split = sample.split(dataset$default.payment.next.month, SplitRatio = 0.7)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Fitting the K Nearest Neighbour algorithm to classify the output as desired. 

KNN_pred = knn(train = training_set[, -24],
               test = test_set[, -24],
               cl = training_set[, 24],
               k = 95,
               prob = TRUE)

# Making the Confusion Matrix
Knn_cm = table(test_set[, 24], KNN_pred)
Knn_cm
confusionMatrix(Knn_cm)
#As we can see from the output, our Accuracy ( Sum of True positive and True Negative / Number of Test Observations) is 80.8%
#Also, our sensitivity is 82.99%. 

#Lets try and fit a Logistic Regression to the same data. 

#################################
#########Logistic Regression#####
#################################
mylogit<-glm(default.payment.next.month~., data = training_set,family = "binomial")

prob_pred<-predict(mylogit,type='response',newdata = test_set[-24])
prob_pred
#As we know, we get probabilities out of the logistic regression model. So in this case, we are keeping of threshold of 50%. Anything below 50% are taken as zero and above it is 1. 

y_pred=ifelse(prob_pred>0.5,1,0)
cm1=table(test_set[,24],y_pred)
confusionMatrix(cm1)

#Lets plot the ROC curve and see which model is best. 
#Logistic

logroc<-predict(mylogit,test_set,type="response")
pred_glm<-prediction(logroc,test_set$default.payment.next.month)
perf_glm <- performance(pred_glm, "tpr", "fpr")
auc_glm <- performance(pred_glm,"auc")
auc_glm <- round(as.numeric(auc_glm@y.values),3)
auc_glm
plot(perf_glm, main = "ROC curves for the models", col='blue')
abline(0,1,col="grey")

#K Nearest Neighbours

KNN_pred=as.numeric(KNN_pred)
#knnroc<-predict(y_pred1,test_set,type="response")
pred_knn<-prediction(KNN_pred,test_set$default.payment.next.month)
perf_knn <- performance(pred_knn, "tpr", "fpr")
auc_knn <- performance(pred_knn,"auc")
auc_knn <- round(as.numeric(auc_knn@y.values),3)
auc_knn
plot(perf_knn, add=TRUE, col='yellow')

legend('right', c("KNN", "Logistic Regression"), fill = c('yellow','blue'), bty='n')

#From the concept of area under curve, and accuracy from confusion matrix we chose Logistic regression as the best model. 