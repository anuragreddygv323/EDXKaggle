# KAGGLE COMPETITION (EDX): Predict whether an iPad is going to sell on ebay

# 'This file contains the 3 best models'

rm(list= ls())
ebaytrain = read.csv('./ebay/ebayiPadTrain.csv', stringsAsFactors = FALSE) # training set
ebaytest = read.csv('./ebay/eBayiPadTest.csv', stringsAsFactors = FALSE) # test set, does not contain ouctome variable

ebaytrain$TotalWords = nchar(ebaytrain$description) # To evaluate whether the extent of the description is a predictor
ebaytest$TotalWords = nchar(ebaytest$description)

for (i in 4:9) { ebaytrain[,i] = as.factor(ebaytrain[,i]) }
for (i in 4:9) { ebaytest[,i] = as.factor(ebaytest[,i])}

ebaytrain = subset(ebaytrain, productline !='iPad 5' & productline !='iPad mini Retina') # the products iPad mini and iPad 5 are not present in test set
ebaytrain$productline = factor(ebaytrain$productline) # to remove the extra factor names

library(caTools)
set.seed(144)
spl = sample.split(ebaytrain$sold, SplitRatio = 0.7)

######################################################################
# First models : Using non-text variables + TotalWord count
#I separate my training set in training and testing to verify the accuracy of my models
# Models are evaluated by the auc ('Area under the curve')

train = subset(ebaytrain, spl == T)
test = subset(ebaytrain, spl == F)
train$description = NULL # remove description column
test$descritpion = NULL
train$UniqueID = NULL # remove ID column
test$UniqueID = NULL

#First model: logistic regression using all the variables
logReg = glm(sold ~. -color-carrier-TotalWords, data = train, family = binomial)
testPred = predict(logReg, newdata = test, type = 'response')
library(ROCR)
ROCRpred = prediction(testPred, test$sold)
auc = as.numeric(performance(ROCRpred, "auc")@y.values)   #0.8644184 & AIC: 1220.5: THIS LOOKS LIKE THE BEST, best auc and smallest AIC

#To submit:
ebaytrain1 = ebaytrain   # So I can re-use the datasets later on the code
ebaytest1 = ebaytest
ebaytrain$description = NULL
ebaytest$description = NULL
ebaytrain$UniqueID = NULL
UniqueID = ebaytest$UniqueID
ebaytest$UniqueID = NULL

logReg = glm(sold ~.-color-carrier-TotalWords, data = ebaytrain, family = binomial)
testPred = predict(logReg, newdata = ebaytest, type = 'response')

submission = data.frame(UniqueID = UniqueID, Probability1 = testPred)
write.csv(submission, './ebay/submission1.csv', row.names=FALSE)

######################################################################
# Second models: random forest using non-text variables + TotalWord count
library(randomForest)
train$sold = as.factor(train$sold)
test$sold = as.factor(test$sold)
train$carrier = NULL
train$color = NULL
test$carrier = NULL
test$color = NULL
set.seed(2000)
ebayRF =  randomForest(sold ~., data = train)
testPred = predict(ebayRF, newdata = test, type = 'prob')[,2]
ROCRpred = prediction(testPred, test$sold)
auc = as.numeric(performance(ROCRpred, "auc")@y.values) #0.8760395

#To submit:
ebaytrain = ebaytrain1
ebaytest = ebaytest1
ebaytrain = cbind(ebaytrain[,2:5], ebaytrain[,8:10], ebaytrain[,12])
names(ebaytrain) = names(train)
ebaytest = cbind(ebaytest[,2:5], ebaytest[,8:9], ebaytest[,11])
names(ebaytest)[7] = 'TotalWords'
ebaytrain$sold = as.factor(ebaytrain$sold)

ebayRF =  randomForest(sold ~., data = ebaytrain)
testPred = predict(ebayRF, newdata = ebaytest, type = 'prob')[,2]

submission = data.frame(UniqueID = UniqueID, Probability1 = testPred)
write.csv(submission, './ebay/submission2.csv', row.names=FALSE)

# From the random forest and logistic regressions it seems that the variables I chose to evaluate (lowprice, 
# good condition, lowstartprice, and nodescription) are not really contributing as predictors of sale.

################################################################
# Fifth model: clustering + then predict (without bag of words)
ebaytrain = ebaytrain1
ebaytest = ebaytest1

ebaytrain = ebaytrain[,-c(1,11)]

for (i in 3:8) {ebaytrain[,i] = as.numeric(ebaytrain[,i])}

train = subset(ebaytrain, spl = T)
test = subset(ebaytrain, spl = F)

#Normalize the data
library(caret)
preproc = preProcess(train)
trainNorm = predict(preproc, train)
testNorm = predict(preproc, test)

trainNorm$sold = NULL
set.seed(88)
kmeansClust = kmeans(trainNorm, centers=2)
tapply(train$biddable, kmeansClust$cluster, mean) # to check in a few variables how the clusters differ
tapply(train$startprice, kmeansClust$cluster, mean)
tapply(train$carrier, kmeansClust$cluster, mean) # we see that the custers differ on the carrier

library(flexclust)
km.kcca = as.kcca(kmeansClust, trainNorm)
clusterTrain = predict(km.kcca) 
testNorm$sold = NULL
clusterTest = predict(km.kcca, newdata=testNorm)

for (i in 3:8) {train[,i] = as.factor(train[,i])}
for (i in 3:8) {test[,i] = as.factor(test[,i])}

train1 = subset(train, clusterTrain == 1)
train2 = subset(train, clusterTrain == 2)
test1 = subset(test, clusterTest == 1)
test2 = subset(test, clusterTest == 2)

#predictions for each cluster in train set: Random Forest
train1$sold = as.factor(train1$sold)
train2$sold = as.factor(train2$sold)
test1$sold = as.factor(test1$sold)
test2$sold = as.factor(test2$sold)

RF1 = randomForest(sold ~ ., data=train1)
RF2 = randomForest(sold ~ ., data=train2)

testPred1 = predict(RF1, newdata = test1, type="response")
testPred2 = predict(RF2, newdata = test2, type="response")

# Pooling the predictions
allPredictions = c(as.numeric(as.character(testPred1)), as.numeric(as.character(testPred2)))
allOutcomes = c(as.numeric(as.character(test1$sold)), as.numeric(as.character(test2$sold)))
ROCRpred = prediction(allPredictions, allOutcomes)
auc = as.numeric(performance(ROCRpred, "auc")@y.values) # 0.9726777 A SUBSTANTIAL INCREASE FROM THE NON CLUSTERED RANDOM FOREST MODEL

# Conclusion: clustering + Random Forest increases substantially the predictive power of the algorithm
# This is the best model to predict whether an iPad will sell on ebay given our datasets
# I did not submit this one to the kaggle competition because I joined a few days before the deadline.

#GENERATE THE SUBMISSION
UniqueID = ebaytest$UniqueID
ebaytest = ebaytest[,-c(1,10)]

for (i in 3:8) {ebaytest[,i] = as.numeric(ebaytest[,i])}

#Normalize the data
trainNorm = predict(preproc, ebaytrain)
testNorm = predict(preproc, ebaytest)

#Make clusters
trainNorm$sold = NULL
set.seed(88)
kmeansClust = kmeans(trainNorm, centers=2)

km.kcca = as.kcca(kmeansClust, trainNorm)
clusterTrain = predict(km.kcca) 
clusterTest = predict(km.kcca, newdata=testNorm)

for (i in 3:8) {ebaytrain[,i] = as.factor(ebaytrain[,i])}
for (i in 3:8) {ebaytest[,i] = as.factor(ebaytest[,i])}

train1 = subset(ebaytrain, clusterTrain == 1)
train2 = subset(ebaytrain, clusterTrain == 2)
test1 = subset(ebaytest, clusterTest == 1)
test2 = subset(ebaytest, clusterTest == 2)
UniqueID1 = UniqueID[clusterTest == 1]
UniqueID2 = UniqueID[clusterTest == 2]

#predictions for each cluster in train set: Random Forest
train1$sold = as.factor(train1$sold)
train2$sold = as.factor(train2$sold)

set.seed(1)
RF1 = randomForest(sold ~ ., data=train1)
RF2 = randomForest(sold ~ ., data=train2)

testPred1 = predict(RF1, newdata = test1, type="response")
testPred2 = predict(RF2, newdata = test2, type="response")

# Pooling the predictions
allPredictions = c(as.numeric(as.character(testPred1)), as.numeric(as.character(testPred2)))
allIDs = c(UniqueID1, UniqueID2)

submission3 = data.frame(UniqueID = allIDs, Probability1 = allPredictions)
submission3 = submission3[order(submission3$UniqueID),]
write.csv(submission3, './ebay/submission3.csv', row.names=FALSE)