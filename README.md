# Credit_card_fraud_detection_using_ML-Rstudio
This project explains in detail, how machine learning can be applied to get better results in fraud detection along with the algorithm, pseudocode, explanation of its implementation and experimentation results

1. Abstract

In this project, a technique for `Credit Card Fraud Detection' is developed. As fraudsters are 
increasing day by day. And fallacious transactions are done by the credit card and there are 
various types of fraud. So to solve this problem we model data sets using machine learning with 
credit card fraud detection. The problem includes modelling past credit card transactions with 
data of the ones that turned out to be fraudulent. This way models are tested individually and 
whatever suits the best is further proceeded. And the foremost goal is to detect fraud by filtering 
the techniques to get better results. In this process, we have focused on analyzing and 
preprocessing data sets as well as deployment of anomaly detection algorithms on the PCA 
transformed data.

2. Introduction

Credit card is generally referred to as a card which belongs to each customer/ cardholder, which 
can be used by their owners to purchase various products, goods and opt for different services 
within their credit card limit. Using a credit card a user can purchase particular products and opt 
for paying at a later period before the next billing cycle.
Credit card frauds can be performed easily without the owners knowledge and involves 
significantly less risk. As each fraudulent transaction appears to be a legitimate transaction, this 
makes detecting more challenging. In 2017, there were 1,579 data breaches and nearly 179 
million records among which Credit card frauds were the most common reported form of fraud 
with 133,015 reports followed by employment or tax-related fraud with 82,051 reports, phone 
fraud with 55045 cases and bank fraud with 5,517 reports as per report released by FTC[6]

According to the US Forum report in 2017, most of the crims on credit cards are related to CNP 
transactions i.e. Credit cards are not present as the security of chip cards have increased. The 
below fig 2.2 shows CNP fraud occurred in each Year.  

For implementing this project Multiple Supervised learning techniques are used. Various 
challenges faced during working with the data set that was obtained through kaggle. These are: 
1) highly unbalanced dataset, 2) large numbers of predictor variables are unlabeled thus it 
becomes difficult to perform different analysis techniques. 3) High variability in the transaction 
amount makes it difficult for predictability.
Different Supervised machine learning algorithms [3] like Decision Trees, Logistic Regression and 
LDA, QDA, KNN, Decision tree and Artificial Neural Network are used to detect fraudulent 
transactions in real-time datasets. Two methods under unbalanced data set perform 
extraordinarily well which are Isolation forest algorithm and local outlier factor algorithm are also 
implemented and their results are compared with other models. The reason for these models to 
work is also explained. The future work will focus on solving the above-mentioned problem. The 
algorithm of the random forest itself should be improved.
Though supervised learning methods can be used, they may fail at certain cases of detecting the 
fraud cases. A model of deep Auto-encoder and restricted Boltzmann machine (RBM) [2] that can 
construct normal transactions to find anomalies from normal patterns. Not only that a hybrid 
method is developed with a combination of Adaboost and Majority Voting methods
