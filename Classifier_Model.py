## ==========================================================================================================
## Script name: Classifier_Model.py
## Purpose of script: On the training set, the classifier of the filtered features is constructed 
##                    by the multiple classifier, and the performance evaluation is performed 
##                    on both cross-validation k-fold test set and independent validation set.
## Author: Kaikun Xu
## Date Updated: 2023-02-28
## ChangeLog: Improve method in parse json file in 2023-02-28
## Copyright (c) Kaikun Xu 2023. All rights reserved.
## ==========================================================================================================
import sys, os, re
import json
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from Feature_Importance import extractFeatureLabel,kFoldSplit

def parseJson(jsonSavePath):
    """Exract the features information of the training set and the test set in json file."""
    with open(jsonSavePath,'r',encoding='utf-8') as jsonFile:
        resDict = json.load(jsonFile)
    # The Json storage format may change the order of samples, resorting index here.
    featureTrain = pd.DataFrame.from_dict(resDict["featureTrain"]).rename_axis(
        index="PatientID",columns="GeneName").sort_index()
    labelTrain = pd.DataFrame.from_dict(resDict["labelTrain"]).rename_axis(
        index="PatientID").sort_index()
    featureValidation = pd.DataFrame.from_dict(resDict["featureValidation"]).rename_axis(
        index="PatientID",columns="GeneName").sort_index()
    labelValidation = pd.DataFrame.from_dict(resDict["labelValidation"]).rename_axis(
        index="PatientID").sort_index()
    return featureTrain,labelTrain,featureValidation,labelValidation

def ridge(featureTrain,labelTrain,featureValidation,labelValidation):
    """Predict disease type and evalute model metrics via Ridge regression and Ridge classifier."""
    # Regression Model
    ridgeRegression = Ridge(alpha=1.0,random_state=12345)
    ridgeRegression.fit(X=featureTrain.values, y=labelTrain.values.reshape(-1))
    labelValidationScore = ridgeRegression.predict(featureValidation.values)
    # Classifier Model, using same random_state to ensure same shuffled data
    ridgeClassifier = RidgeClassifier(alpha=1.0, random_state=12345)
    ridgeClassifier.fit(X=featureTrain.values, y=labelTrain.values.reshape(-1))
    labelValidationPred = ridgeClassifier.predict(featureValidation.values)
    labelValidationTrue = np.array(labelValidation).reshape(-1)
    return labelValidationTrue,labelValidationScore,labelValidationPred

def svm(featureTrain,labelTrain,featureValidation,labelValidation):
    """Predict disease type and evalute model metrics via SVM classifier."""
    estimator = SVC(kernel="rbf",probability=True)
    estimator.fit(X=featureTrain.values, y=labelTrain.values.reshape(-1))
    labelValidationScore = estimator.predict_proba(featureValidation.values)[:,1]
    labelValidationPred = estimator.predict(featureValidation.values)
    labelValidationTrue = np.array(labelValidation).reshape(-1)
    return labelValidationTrue,labelValidationScore,labelValidationPred

def knn(featureTrain,labelTrain,featureValidation,labelValidation):
    """Predict disease type and evalute model metrics via K neighbors classifier."""
    estimator = KNeighborsClassifier(n_neighbors=2)
    estimator.fit(X=featureTrain.values, y=labelTrain.values.reshape(-1))
    labelValidationScore = estimator.predict_proba(featureValidation.values)[:,1]
    labelValidationPred = estimator.predict(featureValidation.values)
    labelValidationTrue = np.array(labelValidation).reshape(-1)
    return labelValidationTrue,labelValidationScore,labelValidationPred

def naiveBayes(featureTrain,labelTrain,featureValidation,labelValidation):
    """Predict disease type and evalute model metrics via Gaussian Naive Bayes classifier."""
    estimator = GaussianNB()
    estimator.fit(X=featureTrain.values, y=labelTrain.values.reshape(-1))
    labelValidationScore = estimator.predict_proba(featureValidation.values)[:,1]
    labelValidationPred = estimator.predict(featureValidation.values)
    labelValidationTrue = np.array(labelValidation).reshape(-1)
    return labelValidationTrue,labelValidationScore,labelValidationPred

def decisionTree(featureTrain,labelTrain,featureValidation,labelValidation):
    """Predict disease type and evalute model metrics via DT classifier."""
    estimator = DecisionTreeClassifier(random_state=12345)
    estimator.fit(X=featureTrain.values, y=labelTrain.values.reshape(-1))
    labelValidationScore = estimator.predict_proba(featureValidation.values)[:,1]
    labelValidationPred = estimator.predict(featureValidation.values)
    labelValidationTrue = np.array(labelValidation).reshape(-1)
    return labelValidationTrue,labelValidationScore,labelValidationPred

def randomForest(featureTrain,labelTrain,featureValidation,labelValidation):
    """Predict disease type and evalute model metrics via random forest classifier."""
    estimator = RandomForestClassifier(bootstrap=True,n_estimators=200,random_state=12345)
    estimator.fit(X=featureTrain.values, y=labelTrain.values.reshape(-1))
    labelValidationScore = estimator.predict_proba(featureValidation.values)[:,1]
    labelValidationPred = estimator.predict(featureValidation.values)
    labelValidationTrue = np.array(labelValidation).reshape(-1)
    return labelValidationTrue,labelValidationScore,labelValidationPred

def estimatorSchedule(featureTrain,labelTrain,featureValidation,labelValidation,classifier="SVM"):
    """This function is used to schedule estimator for classifier."""
    if classifier in ("ridge","Ridge"):
        return ridge(featureTrain,labelTrain,featureValidation,labelValidation)
    elif classifier in ("knn","KNN"):
        return knn(featureTrain,labelTrain,featureValidation,labelValidation)
    elif classifier in ("svm","SVM"):
        return svm(featureTrain,labelTrain,featureValidation,labelValidation)
    elif classifier in ("nb","Naive Bayes"):
        return naiveBayes(featureTrain,labelTrain,featureValidation,labelValidation)
    elif classifier in ("dt","Decision Tree"):
        return decisionTree(featureTrain,labelTrain,featureValidation,labelValidation)
    elif classifier in ("rf","Random Forest"):
        return randomForest(featureTrain,labelTrain,featureValidation,labelValidation)

def classifierKFold(featureTrain,labelTrain,featureValidation,labelValidation,classifier="SVM",randomState=12345):
    """The classifier model is constructed by using 5-fold cross-validation, \
    and predicted scores are calculated on both cross-validation test set and independent validation set:
    
    + Cross-validation test set: merged predicted of five submodels yScore as yScore;
    + Independent validation set: mean value of predicted yScore of five submodels as yScore.
    """
    # K-Fold Test Set
    testFoldResDF = pd.DataFrame()
    for featureTrainFold,labelTrainFold,featureTestFold,labelTestFold in kFoldSplit(
        featureTrain,labelTrain,kfold=5,randomState=randomState):
        labelTestFoldTrue,labelTestFoldScore,labelTestFoldPred = estimatorSchedule(
            featureTrainFold,labelTrainFold,featureTestFold,labelTestFold,classifier=classifier)
        testFoldResSub = pd.DataFrame([labelTestFoldTrue,labelTestFoldScore,labelTestFoldPred],
                                      columns=labelTestFold.index,
                                      index=["yTrue","yScore","yPred"]).transpose()
        testFoldResDF = pd.concat([testFoldResDF,testFoldResSub])
    # Validation Set
    labelValidationScoreList = list()
    validationResDF = pd.DataFrame(labelValidation.values,columns=["yTrue"],index=labelValidation.index)
    for featureTrainFold,labelTrainFold,_,_ in kFoldSplit(
        featureTrain,labelTrain,kfold=5,randomState=randomState):
        _,labelValidationScore,_ = estimatorSchedule(
            featureTrainFold,labelTrainFold,featureValidation,labelValidation,classifier=classifier)
        labelValidationScoreList.append(labelValidationScore)
    validationResDF["yScore"] = np.array(labelValidationScoreList).mean(axis=0)
    validationResDF["yPred"] = pd.cut(validationResDF["yScore"],bins=[-np.inf,0.5,np.inf],labels=[0,1])
    return testFoldResDF,validationResDF

def getClassifierPara(yTrue,yScore,yPred):
    """This function is used to calculate the metrics of the classifier model, \
    including specificity, sensitivity, precision, accuracy, f1-score and auroc"""
    confusionArray = metrics.confusion_matrix(y_true=yTrue, y_pred=yPred,labels=[0,1])
    tn, fp, fn, tp = confusionArray.ravel()
    specificity = tn/(tn+fp) # True negative rate (TNR)
    sensitivity = tp/(tp+fn) # True positive rate (TPR), Recall
    precision = tp/(tp+fp) # Positive predictive value (PPV)
    accuracy = metrics.accuracy_score(y_true=yTrue, y_pred=yPred)
    f1Score = metrics.f1_score(y_true=yTrue, y_pred=yPred)
    auroc = metrics.roc_auc_score(y_true=yTrue, y_score=yScore) # Area under ROC
    return specificity, sensitivity, precision, accuracy, f1Score, auroc

if __name__=="__main__":
    classifierPath = os.path.join(os.getcwd(),"Classifier")
    metricsDF = pd.DataFrame(index=["Specificity", "Sensitivity", "Precision","Accuracy", "F1-Score", "AUROC"],
                             columns=pd.MultiIndex.from_arrays(
                                 [[],[],[]],names=("Dataset","Comparsion","Classifier")))
    for groupName in ["CHB/N","LC/N","HCC/N","LC/CHB","HCC/CHB","HCC/LC"]:
        # Determine the name of the experimental group and control group.
        exp,ctrl = re.split("/",groupName)
        # Load features of the training set and the validation set respectively.
        jsonSavePath = os.path.join(classifierPath,"{0}_{1}_LassoCoef_FeatureInfo.json".format(exp,ctrl))
        featureTrain,labelTrain,featureValidation,labelValidation = parseJson(jsonSavePath)
        # Evaluate the performance of each classifier.
        for classifier in ("Ridge","SVM","KNN","Naive Bayes","Decision Tree","Random Forest"):
            testFoldResDF,validationResDF = classifierKFold(
                featureTrain,labelTrain,featureValidation,labelValidation,classifier=classifier)
            metricsDF[("K-Fold Test Set",groupName,classifier)] = getClassifierPara(
                *testFoldResDF.transpose().values.tolist())
            metricsDF[("Validation Set",groupName,classifier)] = getClassifierPara(
                *validationResDF.transpose().values.tolist())
    metricsDF = metricsDF.sort_index(axis=1).transpose()
    metricsDF.to_csv(os.path.join(classifierPath,"Classifier_Metrics_LASSO.csv"))
metricsDF.loc["Validation Set"].head(10)
