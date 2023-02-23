## ==========================================================================================================
## Script name: Model_Visulization.py
## Purpose of script: This script is used to generate the result plots of the machine learning model, 
## including confusion matrix and ROC curve of K-Fold set and validation set.
## Date Updated: 2023-02-14
## Copyright (c) 2023. All rights reserved.
## ==========================================================================================================
import sys, os, re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from Feature_Importance import extractFeatureLabel
from Classifier_Model import classifierKFold


def confusionHeatmap(yTrue,yPred,exp,ctrl,title):
    """Plot confusion heatmap from given yTrue and yPred."""
    confusionArray = metrics.confusion_matrix(y_true=yTrue, y_pred=yPred,labels=[0,1])
    confusionDF = pd.DataFrame(confusionArray,index=[ctrl,exp],columns=[ctrl,exp])   
    fig,(ax) = plt.subplots(ncols=1,figsize=(3.2,3))
    sns.heatmap(confusionDF, annot=True,ax=ax,cmap="Reds",
                cbar=True,annot_kws={"size": 12},cbar_kws={"shrink": .618})
    ax.set_xlabel("{0}".format("Predicted Label"),fontsize=12,weight="bold")
    ax.set_ylabel("{0}".format("True Label"),fontsize=12,weight="bold")
    for item in ([ax.xaxis.label, ax.yaxis.label]): item.set_fontsize(16)
    for item in (ax.get_xticklabels()+ax.get_yticklabels()): item.set_fontsize(14) 
    plt.suptitle(title,fontsize=16,weight="bold",y=1.05)
    plt.tight_layout()
    plt.close(fig)
    return fig,ax
    
def rocDataPreparation(yTrue,yScore):
    """Prepare data for drawing ROC, including fpr(x), trp(y) and auroc(metrics)"""
    fpr, tpr, _ = metrics.roc_curve(y_score = yScore, y_true= yTrue)
    auroc = metrics.auc(fpr, tpr)
    return fpr, tpr, auroc

def emptyROC():
    """Declare an empty ROC curve."""
    fig,(ax) = plt.subplots(ncols=1,nrows=1,figsize=(4.5,4))
    sns.despine(top=True, right=True, left=False, bottom=False) 
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    return fig,(ax)

def groupROC(fpr,tpr,auroc,palette,group,ax):
    """Add ROC for model in data set"""
    ax.plot(fpr, tpr, lw=2.5, color=palette[group], label="{0}: {1:.3f}".format(group,auroc))

def formatROC(fig,ax):
    """Declare an empty ROC curve."""
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=10,title_fontsize=10,loc="lower right",shadow=True,frameon=True)
    for item in ([ax.xaxis.label, ax.yaxis.label]): item.set_fontsize(16)
    for item in (ax.get_xticklabels()+ax.get_yticklabels()): item.set_fontsize(14)
    plt.tight_layout()
    plt.close(fig)

if __name__=="__main__":
    classifier = "SVM"
    classifierPath = os.path.join(os.getcwd(),"Classifier")
    # Load AFP parameter of CHB/LC/HCC patients.
    afpDF = pd.read_excel(os.path.join(classifierPath,"Ditan AFP information.xlsx"),index_col=[0])
    afpDF = afpDF.rename(columns={"AFP(ng/mL)":"AFP"}).drop(columns=["Unnamed: 3","DiseaseType"])
    afpDF["AFP"] = afpDF["AFP"].replace({">2000":1638.3}).astype(float)
    afpDF["AFP_Pred"] = pd.cut(afpDF["AFP"],bins=[-np.inf,8.78,np.inf],labels=[0,1])
    for groupName in ["HCC/CHB","HCC/LC","HCC/N","LC/CHB","LC/N","CHB/N"]:
        # groupName = "LC/CHB"
        # Determine the name of the experimental group and control group.
        exp,ctrl = re.split("/",groupName)
        # Load data of the training set and the validation set respectively.
        trainingDF = pd.read_csv(
            os.path.join(classifierPath,"Training-Set-DIAMS-Integrating-{0}.csv".format(
            "&".join(re.split("/",groupName)))),header=[0,1],index_col=[0]).transpose()
        validationDF = pd.read_csv(
            os.path.join(classifierPath,"Validation-Set-DIAMS-Integrating-{0}.csv".format(
            "&".join(re.split("/",groupName)))),header=[0,1],index_col=[0]).transpose()
        # Normalize features by removing the mean and scaling to unit variance.
        scaler = StandardScaler()
        trainingDF = pd.DataFrame(scaler.fit_transform(np.log10(trainingDF)),
                                  index=trainingDF.index,columns=trainingDF.columns)
        validationDF = pd.DataFrame(scaler.fit_transform(np.log10(validationDF)),
                                    index=validationDF.index,columns=validationDF.columns)
        # Extract candidate proteins (n-highest importance) from feature importance file.
        featureImportanceDF = pd.read_csv(
            os.path.join(classifierPath,"{0}_{1}_Lasso_Coef.csv".format(exp,ctrl)),
            index_col=[0],header=[0,1])
        candidiateList = featureImportanceDF[featureImportanceDF[("Total","Coef_Per")]>0.1].index
        trainingDF = trainingDF[candidiateList]; validationDF = validationDF[candidiateList]
        # Extract features and labels from training set and validation set.
        assert (trainingDF.columns.tolist() == validationDF.columns.tolist()
               ),"The features of the training set and validation set do not match."
        featureTrain,labelTrain = extractFeatureLabel(trainingDF,exp=exp,ctrl=ctrl)
        featureValidation,labelValidation = extractFeatureLabel(validationDF,exp=exp,ctrl=ctrl)
        # Calculate pred score and pred label of SVM model
        testFoldResDF,validationResDF = classifierKFold(
            featureTrain,labelTrain,featureValidation,labelValidation,classifier=classifier)
        # Plot confusion matrix of SVM model
        testFoldTrue,testFoldScore,testFoldPred = testFoldResDF.transpose().values.tolist()
        labelValidationTrue,labelValidationScore,labelValidationPred = validationResDF.transpose().values.tolist()
        figTest,axTest = confusionHeatmap(
            yTrue=testFoldTrue,yPred=testFoldPred,exp=exp,ctrl=ctrl,
            title="K-Fold Test Set: {0}/{1}".format(exp,ctrl))
        figTest.savefig(os.path.join(
            classifierPath,"CM_KFoldTestSet_{0}_{1}.pdf".format(exp,ctrl)),bbox_inches="tight")
        figValidation,axValidation = confusionHeatmap(
            yTrue=labelValidationTrue,yPred=labelValidationPred,exp=exp,ctrl=ctrl,
            title="Validation Set: {0}/{1}".format(exp,ctrl))
        figValidation.savefig(os.path.join(
            classifierPath,"CM_ValidationSet_{0}_{1}.pdf".format(exp,ctrl)),bbox_inches="tight")
        # Plot ROC of SVM model
        colorDict ={f"{classifier} (K-Fold)":"tab:orange",f"{classifier} (Validation)":"tab:red",
                    "AFP (Validation)":"gray"}
        fig,ax = emptyROC()
        # fpr,tpr,auroc = rocDataPreparation(yTrue=testFoldResDF["yTrue"],yScore=testFoldResDF["yScore"])
        # groupROC(fpr,tpr,auroc,palette=colorDict,group="K-Fold",ax=ax); formatROC(fig,ax)
        validationResDF = pd.merge(left=validationResDF,right=afpDF,left_index=True,right_index=True,how="left")
        fpr,tpr,auroc = rocDataPreparation(yTrue=validationResDF["yTrue"],yScore=validationResDF["yScore"])
        groupROC(fpr,tpr,auroc,palette=colorDict,group=f"{classifier} (Validation)",ax=ax); formatROC(fig,ax)
        if groupName in ["HCC/CHB","HCC/LC","LC/CHB"]:
            fpr,tpr,auroc = rocDataPreparation(yTrue=validationResDF["yTrue"],yScore=validationResDF["AFP_Pred"])
            groupROC(fpr,tpr,auroc,palette=colorDict,group="AFP (Validation)",ax=ax); formatROC(fig,ax)
        ax.set_title(groupName,fontsize=16,weight="bold")
        fig.savefig(
            os.path.join(classifierPath,"ROC_ValidationSet_{0}_{1}.pdf".format(exp,ctrl)),bbox_inches="tight")
