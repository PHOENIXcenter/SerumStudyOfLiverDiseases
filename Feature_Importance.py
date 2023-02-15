## ==========================================================================================================
## Script name: Feature_Importance.py
## Purpose of script: Calculate feature importance of each protein feature via RF or LASSO
## Date Updated: 2023-02-03
## Copyright (c) 2023. All rights reserved.
## ==========================================================================================================
import sys, os, re
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeClassifier, Lasso
from sklearn.ensemble import RandomForestClassifier

def extractFeatureLabel(dataframe,exp=False,ctrl=False):
    """This function is used to divide the training set and testing set of dataframe.
    
    + Input:
    | DiseaseType | PatientID | A2M      | ...      | APOA2    |
    | :---------: | :-------: | :------: | :------: | :------: |
    | N           | N-N392    | 7.102001 | ...      | 6.450542 |
    | N           | N-N398    | 6.502496 | ...      | 6.545598 |
    | ...         | ...       | ...      | ...      | ...      |
    | HCC         | HCC-236   | 6.520594 | ...      | 6.526614 |
    | HCC         | HCC-238   | 6.517217 | ...      | 6.457601 |
    + Output:
    # Features:
    | PatientID | A2M      | ...      | APOA2    |
    | :-------: | :------: | :------: | :------: |
    | N-N392    | 7.102001 | ...      | 6.450542 |
    | N-N398    | 6.502496 | ...      | 6.545598 |
    | ...       | ...      | ...      | ...      |
    | HCC-236   | 6.520594 | ...      | 6.526614 |
    | HCC-238   | 6.517217 | ...      | 6.457601 |    
    # Labels:
    | PatientID | DiseaseType |
    | :-------: | :---------: |
    | N-N392    | 0           |
    | N-N398    | 0           |
    | ...       | ...         |
    | HCC-236   | 1           |
    | HCC-238   | 1           |
    """
    assert all([exp,ctrl]),"The names of the experimental and control groups were not specified."
    group, sample = dataframe.index.names
    features = dataframe.reset_index([group],drop=True)
    labels = dataframe.index.to_frame().reset_index([group],drop=True).drop(
        columns=[sample]).replace({ctrl:0,exp:1})
    return features,labels

def KFoldSplit(features,labels,kfold=5,randomState=0):
    """This function is used to split features and labels with index criteria."""
    kf = KFold(n_splits=kfold,shuffle=True,random_state=randomState)
    splitData = list()
    for (trainIndex, testIndex) in kf.split(labels):
        featureTrain,featureTest = features.iloc[trainIndex], features.iloc[testIndex]
        labelTrain,labelTest = labels.iloc[trainIndex], labels.iloc[testIndex]
        splitData.append([featureTrain,labelTrain,featureTest,labelTest])
    return splitData

def lassoCoef(features,labels):
    """Return coef of each feature via lasso model."""
    lassoModel = Lasso(alpha=0.1)
    lassoModel.fit(X=features.values,y=labels.values.reshape(-1))
    return lassoModel.coef_

def LassoCoefLoop(features,labels,kfold=5,epochMax=100):
    """Select the most important features for classification by lasso modeling with the down-sampled data."""
    coefDF = pd.DataFrame()
    for randomState in np.arange(0,epochMax,1):
        coefState = pd.DataFrame(index=features.columns).rename_axis(columns=["Fold"])
        fold = 1
        for featureTrain,labelTrain,_,_ in KFoldSplit(features,labels,kfold=kfold,randomState=randomState):
            # Random select samples k times in each loop by k-fold split;
            # Only k-1/k of the training set is used to assess the importance of features.
            coefState["Fold_{0}".format(fold)] = lassoCoef(features=featureTrain,labels=labelTrain)
            fold = fold + 1
        coefSub = pd.concat([coefState],names=["State"],keys=["State_{0}".format(randomState)],axis=1)
        coefDF = pd.concat([coefDF,coefSub],axis=1)
    # Remove features with 0 weights in all submodels.
    coefDF = coefDF.replace({0:np.nan}).dropna(how="all")
    # Calculate the average coefficient (skip na), number and percent of coefficient across all submodels.
    coefDF[("Total","Coef_Average")],coefDF[("Total","Coef_Num")],coefDF[("Total","Coef_Per")] = coefDF.mean(
        axis=1),coefDF.notnull().sum(axis=1),coefDF.notnull().sum(axis=1)/coefDF.shape[1]
    coefDF = coefDF.sort_values(by=("Total","Coef_Average"),ascending=False)
    return coefDF

def extractFeatures(trainingSetPath,validationSetPath,featureImportanceDF,exp,ctrl,perThresh=0.1):
    """According to the percent of Lasso coefficient among the submodels,/
    the features of training set and validation set are extracted and saved as dict.
    """
    # Extract candidate proteins (n-highest importance) from feature importance file.
    candidiateList = featureImportanceDF[featureImportanceDF[("Total","Coef_Per")]>perThresh].index
    # Load data of the training set and the validation set respectively.
    trainingDF = pd.read_csv(trainingSetPath,header=[0,1],index_col=[0]).transpose()
    validationDF = pd.read_csv(validationSetPath,header=[0,1],index_col=[0]).transpose()
    # Normalize features by removing the mean and scaling to unit variance.
    scaler = StandardScaler()
    trainingDF = pd.DataFrame(scaler.fit_transform(np.log10(trainingDF)),
                              index=trainingDF.index,columns=trainingDF.columns)
    validationDF = pd.DataFrame(scaler.fit_transform(np.log10(validationDF)),
                                index=validationDF.index,columns=validationDF.columns)
    # Generate candidiate features matrix
    trainingDF = trainingDF[candidiateList]; validationDF = validationDF[candidiateList]
    # Extract features and labels from training set and validation set.
    assert (trainingDF.columns.tolist() == validationDF.columns.tolist()
           ),"The features of the training set and validation set do not match."
    featureTrain,labelTrain = extractFeatureLabel(trainingDF,exp=exp,ctrl=ctrl)
    featureValidation,labelValidation = extractFeatureLabel(validationDF,exp=exp,ctrl=ctrl)
    # Save feature information as dict
    resDict = {"featureTrain":featureTrain.to_dict(),"labelTrain":labelTrain.to_dict(),
               "featureValidation":featureValidation.to_dict(),"labelValidation":labelValidation.to_dict()}
    return resDict


if __name__=="__main__":
    classifierPath = os.path.join(os.getcwd(),"Classifier")
    for groupName in ["CHB/N","HCC/CHB","HCC/LC","HCC/N","LC/CHB","LC/N"]:
        # Determine the name of the experimental group and control group.
        exp,ctrl = re.split("/",groupName)
        trainingSetPath = os.path.join(classifierPath,"Training-Set-DIAMS-Integrating-{0}.csv".format("&".join(re.split("/",groupName))))
        validationSetPath = os.path.join(classifierPath,"Validation-Set-DIAMS-Integrating-{0}.csv".format("&".join(re.split("/",groupName))))
        trainingDF = pd.read_csv(trainingSetPath,header=[0,1],index_col=[0]).transpose()
        # Normalize features by removing the mean and scaling to unit variance.
        scaler = StandardScaler()
        trainingDF = pd.DataFrame(scaler.fit_transform(np.log10(trainingDF)),index=trainingDF.index,columns=trainingDF.columns)
        # Extract features and labels from origin dataframe.
        featureTrain,labelTrain = extractFeatureLabel(trainingDF,exp=exp,ctrl=ctrl)
        # Calculate feature importance and save as csv file.
        featureImportanceDF = LassoCoefLoop(featureTrain,labelTrain,kfold=5,epochMax=100)
        featureImportanceDF.to_csv(os.path.join(classifierPath,"{0}_{1}_Lasso_Coef.csv".format(exp,ctrl)),na_rep="NA") 
        # Extract the features of training set and validaiton set and save dict as json file.
        resDict = extractFeatures(trainingSetPath=trainingSetPath,validationSetPath=validationSetPath,
                                 featureImportanceDF=featureImportanceDF,exp=exp,ctrl=ctrl,perThresh=0.1)
        jsonSavePath = os.path.join(classifierPath,"{0}_{1}_LassoCoef_FeatureInfo.json".format(exp,ctrl))
        with open (jsonSavePath,"w") as jsonFile:
            json.dump(resDict,jsonFile)
