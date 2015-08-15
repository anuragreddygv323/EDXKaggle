# Predicting which iPads on ebay will be sold
###Kaggle competition open to students of the MITx free online course 15.071x on edX- The Analytics Edge.



A training and testing set were provided by [Kaggle](https://www.kaggle.com/solegalli/results): eBayiPadTrain.csv with 1861 listings, and eBayiPadTest.csv with 798 listings. 

The models should predict whether an iPad would be sold (1) or not (0). This constituted the dependent variable, which was provided for the training but not for the testing set.

Within the independent variables we find:

* description = A text description of the product provided by the seller

* biddable = Whether this is an auction (1) or a sale with a fixed price (0)

* startprice = The start price (in US Dollars) of the item

* condition = Whether the product is new, used, etc

* cellular = Whether the iPad has cellular connectivity (1) or not (0)

* carrier = The company that provides the service to the iPad (Verizon, AT&T, etc)

* color = The color of the iPad

* storage = The iPad's storage capacity (in gigabytes)

* productline = The name of the product being sold (for example iPad3, iPad4, iPad mini, etc)

**In this repository you will find:**

* The original datasets provided in Kaggle (ebayiPadTrain.csv and ebayiPadTest.csv)

* The code containing the different models I built (ebayModels.R). I used logistic regression, random forest trees or a combination of clustering + logistic regression or random forests. The code was written in R

* The top 5 models in terms of accuracy and predictive power (Top5Models.csv)

* Graphs:

     + Accuracy of the different predictive models (AccuracyModels.png)
  
     + Area under the Receiver Operator Characteristic (ROC) curve for the different Models (aucModels.png)
  
     + Composition of the clusters for the predictive models in which clustering was used (CarriersInClusters.png and ProductsInClusters.png)
     
     + Receiver Operator Charactristic curve for the Random Forest model number 4, which shows the biggest area under the ROC curve and predicts whether an iPad will sell with high accuracy 

**For more detail on the development of the models and the outcome of the code please visit this [link](www.kaggle.com/solegalli/EDXKaggle/blob/master/Development.Rmd)**
