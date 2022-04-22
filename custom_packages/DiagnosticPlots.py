import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, matthews_corrcoef, roc_auc_score, roc_curve

###############################
    
# Feature Importance

###############################
def feature_importance(model, column_names):

    feature_importance=pd.DataFrame({'Model':model.feature_importances_},index=column_names)
    feature_importance.sort_values(by='Model',ascending=True,inplace=True)

    index = np.arange(len(feature_importance))
    fig, ax = plt.subplots(figsize=(12,8))
    rfc_feature=ax.barh(index,feature_importance['xgboost_model'],0.4,color='rebeccapurple',label='Model')
    ax.set(yticks=index+0.4,yticklabels=feature_importance.index)

    ax.legend()
    plt.show()



###############################
    
# Confusion Matrix

###############################
def confusion_matrix(target, prediction):    
    # Create data for a confusion matrix 
    confusion_matrix_data = pd.crosstab(target, prediction)
    # Confusion Matrix
    sns.heatmap(confusion_matrix_data, cmap='PuOr', annot=True,fmt=".1f",annot_kws={'size':16})



###############################
    
# Plot ROC Curve

###############################
def roc_curve(probabilities, target):
    # Compute AUC score
    area_under_curve = roc_auc_score(target, probabilities)

    # Compute other metrics
    false_positive_rate, true_positive_rate, thresholds = roc_curve(target, probabilities)

    # Plot ROC Curve
    plt.figure(figsize=(12, 7))
    plt.plot(false_positive_rate, true_positive_rate, label=f'AUC = {area_under_curve:.2f}', color = 'rebeccapurple')
    plt.plot([0, 1], [0, 1], color='gold', linestyle='--', label='Baseline')
    plt.title('ROC Curve', size=20)
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.legend()



###############################
    
# Categorical Accuracy Table

###############################
def categorical_accuracy_table(target, prediction):

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        target, prediction, average='weighted')
    accuracy = accuracy_score(prediction, target)
    mcc = matthews_corrcoef(prediction, target)

    measures = ['Accuracy','F1_Score','Precision', 'Recall', 'MCC']
    scores = [accuracy, f1_score, precision, recall, mcc]
    
    accuracytable = pd.DataFrame({'Measure': measures, 'Value': scores})
    
    return accuracytable




    
###############################
    
# FUTURE IMPROVEMENTS

###############################

#sns.heatmap(confusion_matrix_data/np.sum(confusion_matrix_data), cmap='Blues', annot=labels,fmt=".2f",annot_kws={'size':16})
# Add a mask argument to block a row and a column
# Calculate Precision and Recall, Row and Column Totals
# Shows how to do it all below
# https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea