# KAGGLE COMPETITION (EDX): Predict whether an iPad is going to sell on ebay

'This file contains models and thoughts I developed while in the competition'

rm(list= ls())
ebaytrain = read.csv('./ebay/ebayiPadTrain.csv', stringsAsFactors = FALSE) # training set
ebaytest = read.csv('./ebay/eBayiPadTest.csv', stringsAsFactors = FALSE) # test set, does not contain ouctome variable

ebaytrain$TotalWords = nchar(ebaytrain$description) # To evaluate whether the extent of the description is a predictor
ebaytest$TotalWords = nchar(ebaytest$description)

ebaytrain$lowstartprice = ifelse(ebaytrain$startprice <= 242, 1, 0) # To evaluate whether low price is a good predictor
ebaytest$lowstartprice = ifelse(ebaytest$startprice <= 242, 1, 0)

ebaytrain$nodescription = ifelse(nchar(ebaytrain$description)==0,1,0) # To evaluate whether not having a description is a good predictor
ebaytest$nodescription = ifelse(nchar(ebaytest$description)==0,1,0)

ebaytrain$goodcon = ifelse(grepl("good condition",ebaytrain$description,fixed=TRUE), 1, 0) # To evaluate whether descriptions including "good condition" are good predictors
ebaytest$goodcon = ifelse(grepl("good condition",ebaytest$description,fixed=TRUE), 1, 0)

for (i in 4:9) { ebaytrain[,i] = as.factor(ebaytrain[,i]) }
for (i in 4:9) { ebaytest[,i] = as.factor(ebaytest[,i])}

ebaytrain = subset(ebaytrain, productline !='iPad 5' & productline !='iPad mini Retina') # the products iPad mini and iPad 5 are not present in test set
ebaytrain$productline = factor(ebaytrain$productline) # to remove the extra factor names

library(tm)
library(SnowballC)
# analyse text of training and testing sets together
CorpusDescription = Corpus(VectorSource(c(ebaytrain$description, ebaytest$description)))
CorpusDescription = tm_map(CorpusDescription, content_transformer(tolower), lazy=TRUE)
CorpusDescription = tm_map(CorpusDescription, PlainTextDocument, lazy=TRUE)
CorpusDescription = tm_map(CorpusDescription, removePunctuation, lazy=TRUE)
CorpusDescription = tm_map(CorpusDescription, removeWords, stopwords("english"), lazy=TRUE)
CorpusDescription = tm_map(CorpusDescription, stemDocument, lazy=TRUE)

dtm = DocumentTermMatrix(CorpusDescription)
sparse = removeSparseTerms(dtm, 0.99) # I played a lot with this number, did not really affect the predictions
DescriptionWords = as.data.frame(as.matrix(sparse))
colnames(DescriptionWords) = make.names(colnames(DescriptionWords))

# now separate the training and testing sets
DescriptionWordsTrain = head(DescriptionWords, nrow(ebaytrain))
DescriptionWordsTest = tail(DescriptionWords, nrow(ebaytest))

ebaytrain$WordCount = rowSums(DescriptionWordsTrain)
ebaytest$WordCount = rowSums(DescriptionWordsTest)

newtrain = cbind(ebaytrain, DescriptionWordsTrain)
newtest = cbind(ebaytest, DescriptionWordsTest)

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
logReg = glm(sold ~., data = train, family = binomial)
testPred = predict(logReg, newdata = test, type = 'response')
library(ROCR)
ROCRpred = prediction(testPred, test$sold)
auc = as.numeric(performance(ROCRpred, "auc")@y.values)  #0.860241 & AIC: 1236.6

#removing some non-significant variables
logReg = glm(sold ~. -color-carrier-TotalWords, data = train, family = binomial)
testPred = predict(logReg, newdata = test, type = 'response')
library(ROCR)
ROCRpred = prediction(testPred, test$sold)
auc = as.numeric(performance(ROCRpred, "auc")@y.values)   #0.8604362

logReg = glm(sold ~. -color-carrier-TotalWords-goodcon-nodescription-lowstartprice, data = train, family = binomial)
testPred = predict(logReg, newdata = test, type = 'response')
library(ROCR)
ROCRpred = prediction(testPred, test$sold)
auc = as.numeric(performance(ROCRpred, "auc")@y.values)   #0.8644184 & AIC: 1220.5: THIS LOOKS LIKE THE BEST, best auc and smallest AIC

logReg = glm(sold ~. -color-carrier-goodcon-nodescription-lowstartprice-WordCount, data = train, family = binomial)
testPred = predict(logReg, newdata = test, type = 'response')
library(ROCR)
ROCRpred = prediction(testPred, test$sold)
auc = as.numeric(performance(ROCRpred, "auc")@y.values) #0.8629088 & AIC: 1222.2

#All Logistic regression models show  roughly the same auc
# I used the one before last for submission to the kaggle competition.

#To submit:
ebaytrain1 = ebaytrain   # So I can re-use the datasets later on the code
ebaytest1 = ebaytest
ebaytrain$description = NULL
ebaytest$description = NULL
ebaytrain$UniqueID = NULL
UniqueID = ebaytest$UniqueID
ebaytest$UniqueID = NULL

logReg = glm(sold ~.-color-carrier-TotalWords-goodcon-nodescription-lowstartprice, data = ebaytrain, family = binomial)
testPred = predict(logReg, newdata = ebaytest, type = 'response')

submission = data.frame(UniqueID = UniqueID, Probability1 = testPred)
write.csv(submission, './ebay/submission.csv', row.names=FALSE)

######################################################################
# Second models: random forest using non-text variables + TotalWord count
library(randomForest)
train$sold = as.factor(train$sold)
test$sold = as.factor(test$sold)
set.seed(144)
ebayRF =  randomForest(sold ~., data = train)
testPred = predict(ebayRF, newdata = test, type = 'prob')[,2]
ROCRpred = prediction(testPred, test$sold)
auc = as.numeric(performance(ROCRpred, "auc")@y.values) #0.8725974

ebayRF$importance # we see that goodcon and nodescription do not contribute much to decrease the impurity

train2=train
train2$goodcon = NULL
train2$nodescription = NULL
test2 = test
test2$goodcon = NULL
test2$nodescription = NULL
set.seed(144)
ebayRF =  randomForest(sold ~., data = train2)
testPred = predict(ebayRF, newdata = test2, type = 'prob')[,2]
ROCRpred = prediction(testPred, test2$sold)
auc = as.numeric(performance(ROCRpred, "auc")@y.values) #0.8763908 

train2$lowstartprice = NULL
train2$carrier = NULL
train2$color = NULL
test2$lowstartprice = NULL
test2$carrier = NULL
test2$color = NULL
set.seed(2000)
ebayRF =  randomForest(sold ~., data = train2)
testPred = predict(ebayRF, newdata = test2, type = 'prob')[,2]
ROCRpred = prediction(testPred, test2$sold)
auc = as.numeric(performance(ROCRpred, "auc")@y.values) #0.8685306

train2$WordCount = NULL
test2$WordCount = NULL
set.seed(2000)
ebayRF =  randomForest(sold ~., data = train2)
testPred = predict(ebayRF, newdata = test2, type = 'prob')[,2]
ROCRpred = prediction(testPred, test2$sold)
auc = as.numeric(performance(ROCRpred, "auc")@y.values) # 0.8760395

# All the random forest trees have aproximately the same auc. I choose the last one for submission:
# It includes the variables 'biddable', 'startprice', 'condition', 'cellular', 'storage', 'productline' and 'TotalWords'
# THIS WAS THE ONE I USED IN THE KAGGLE COMPETITION

#To submit:
ebaytrain = ebaytrain1
ebaytest = ebaytest1
ebaytrain = cbind(ebaytrain[,2:5], ebaytrain[,8:10], ebaytrain[,12])
names(ebaytrain) = names(train2)
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
# MODELS CONSIDERING TEXT VARIABLE 'DESCRIPTION'

train = subset(newtrain, spl == T)
test = subset(newtrain, spl == F)
train$UniqueID = NULL
test$UniqueID = NULL
train$description = NULL
test$description = NULL

#RF
train$sold = as.factor(train$sold)
test$sold = as.factor(test$sold)

set.seed(2000)
ebayRF = randomForest(sold ~., data = train)
testPred = predict(ebayRF, newdata = test, type = 'prob')[,2]
ROCRpred = prediction(testPred, test$sold)
auc = as.numeric(performance(ROCRpred, "auc")@y.values) #0.8606574 

ebayRF$importance # we see that the different words contribute very little to decreasing the impurity
# therefore, the bag of words does not improve the accuracy of our model over the ones evaluated above

######################################################################################
# Fourth model; different models for 'auction' and 'buy it now'
ebaytrain = ebaytrain1
ebaytest = ebaytest1

ebaytrainbid = subset(ebaytrain, biddable == 1)
ebaytrainbuyitnow = subset(ebaytrain, biddable == 0)

set.seed(200)
spl2 = sample.split(ebaytrainbid$sold, SplitRatio = 0.7)
trainbid = subset(ebaytrainbid, spl2 == TRUE)
testbid = subset(ebaytrainbid, spl2 == FALSE)

spl3 = sample.split(ebaytrainbuyitnow$sold, SplitRatio = 0.7)
trainBIN = subset(ebaytrainbuyitnow, spl3 == T)
testBIN = subset(ebaytrainbuyitnow, spl3 == F)

trainbid$sold = as.factor(trainbid$sold)
testbid$sold = as.factor(testbid$sold)
trainbid$description = NULL
trainbid$UniqueID = NULL
trainbid$biddable = NULL
testbid$description = NULL
testbid$UniqueID = NULL
testbid$biddable = NULL
set.seed(144)
ebayRF =  randomForest(sold ~., data = trainbid)
testPred = predict(ebayRF, newdata = testbid, type = 'prob')[,2]
ROCRpred = prediction(testPred, testbid$sold)
auc = as.numeric(performance(ROCRpred, "auc")@y.values) # 0.8443518

ebayRF$importance # we see that the highly contributing variables are: startprice, condition, storrage, productline, TotalWords, lowstartprice

#remove remaining variables and make a second random forest model
trainbid$cellular = NULL
trainbid$carrier = NULL
trainbid$color = NULL
trainbid$nodescription = NULL
trainbid$goodcon = NULL
trainbid$WordCount = NULL
testbid$cellular = NULL
testbid$carrier = NULL
testbid$color = NULL
testbid$nodescription = NULL
testbid$goodcon = NULL
testbid$WordCount = NULL

set.seed(3)
ebayRF =  randomForest(sold ~., data = trainbid)
testPredbid = predict(ebayRF, newdata = testbid, type = 'prob')[,2]
ROCRpred = prediction(testPredbid, testbid$sold)
auc = as.numeric(performance(ROCRpred, "auc")@y.values) #0.8428432

trainBIN$sold = as.factor(trainBIN$sold)
testBIN$sold = as.factor(testBIN$sold)
trainBIN$description = NULL
trainBIN$UniqueID = NULL
trainBIN$biddable = NULL
testBIN$description = NULL
testBIN$UniqueID = NULL
testBIN$biddable = NULL

set.seed(144)
ebayRF =  randomForest(sold ~., data = trainBIN)
testPred = predict(ebayRF, newdata = testBIN, type = 'prob')[,2]
ROCRpred = prediction(testPred, testBIN$sold)
auc = as.numeric(performance(ROCRpred, "auc")@y.values) # 0.6461538

ebayRF$importance # we see that the variables carrier, color are important, and not so much the lowstartprice

trainBIN$cellular = NULL
trainBIN$lowstartprice = NULL
trainBIN$nodescription = NULL
trainBIN$WordCount = NULL
testBIN$cellular = NULL
testBIN$lowstartprice = NULL
testBIN$nodescription = NULL
testBIN$WordCount = NULL

set.seed(144)
ebayRF =  randomForest(sold ~., data = trainBIN)
testPredBIN = predict(ebayRF, newdata = testBIN, type = 'prob')[,2]
ROCRpred = prediction(testPredBIN, testBIN$sold)
auc = as.numeric(performance(ROCRpred, "auc")@y.values) # 0.6311218

allPredictions = c(testPredbid, testPredBIN)
allOutcomes = c(as.numeric(as.character(testbid$sold)), as.numeric(as.character(testBIN$sold)))
ROCRpred = prediction(allPredictions, allOutcomes)
auc = as.numeric(performance(ROCRpred, "auc")@y.values) # 0.8564029 

# From these analyses we see that separating into biddable and non biddable does not really improve the predictive power

#####################################################################################
# Fifth model: clustering + then predict (without bag of words)
ebaytrain = ebaytrain1
ebaytest = ebaytest1

ebaytrain = ebaytrain[,-c(1,11,13,14,15,16)]

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

#predictions for each cluster in train set: logistic regression
logReg1 = glm(sold ~ ., data=train1, family=binomial)
logReg2 = glm(sold ~ ., data=train2, family=binomial)

testPred1 = predict(logReg1, newdata = test1, type="response")
testPred2 = predict(logReg2, newdata = test2, type="response")

# Pooling the predictions
allPredictions = c(testPred1, testPred2)
allOutcomes = c(test1$sold, test2$sold)
ROCRpred = prediction(allPredictions, allOutcomes)
auc = as.numeric(performance(ROCRpred, "auc")@y.values) # 0.8773508 a modest increase over our previous logistic regression without clustering

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
ebaytest = ebaytest[,-c(1,10,12,13,14,15)]

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

###############################################################################
# Sixth model: clustering + predict (with bag of words)

newtrain = newtrain[,-c(1,11,13,14,15,16)]
newtest = newtest[,-c(1,10,12,13,14,15)]

for (i in 3:8) {newtrain[,i] = as.numeric(newtrain[,i])}

train = subset(newtrain, spl = T)
test = subset(newtrain, spl = F)

#Normalize the data
preproc = preProcess(train)
trainNorm = predict(preproc, train)
testNorm = predict(preproc, test)

trainNorm$sold = NULL
set.seed(88)
kmeansClust = kmeans(trainNorm, centers=2)
tapply(train$biddable, kmeansClust$cluster, mean)

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

set.seed(99)
RF1 = randomForest(sold ~ ., data=train1)
RF2 = randomForest(sold ~ ., data=train2)

testPred1 = predict(RF1, newdata = test1, type="response")
testPred2 = predict(RF2, newdata = test2, type="response")

# Pooling the predictions
allPredictions = c(as.numeric(as.character(testPred1)), as.numeric(as.character(testPred2)))
allOutcomes = c(as.numeric(as.character(test1$sold)), as.numeric(as.character(test2$sold)))
ROCRpred = prediction(allPredictions, allOutcomes)
auc = as.numeric(performance(ROCRpred, "auc")@y.values) # 0.8849478  slight improvement

#Conclusion: the bag of words does not really contribute to the predictive power of our algorithm
