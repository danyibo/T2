import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from MeDIT.Visualization import Imshow3DArray
from Tool.DataProcress import get_array_from_path
from sklearn.metrics import roc_curve, roc_auc_score


def get_plot(pred_list, label_list, name_list='', store_path='', is_show=True, fig=plt.figure()):
    '''
    To Draw the ROC curve.
    :param pred_list: The list of the prediction.
    :param label_list: The list of the label.
    :param name_list: The list of the legend name.
    :param store_path: The store path. Support jpg and eps.
    :return: None

    Apr-28-18, Yang SONG [yang.song.91@foxmail.com]
    '''
    if not isinstance(pred_list, list):
        pred_list = [pred_list]
    if not isinstance(label_list, list):
        label_list = [label_list]
    if not isinstance(name_list, list):
        name_list = [name_list]
    for index in range(len(pred_list)):
        fpr, tpr, threshold = roc_curve(label_list[index], pred_list[index])
        auc = roc_auc_score(label_list[index], pred_list[index])
        name_list[index] = name_list[index] + (' (AUC = %0.3f)' % auc)
        return fpr, tpr, auc


def show_roc(test_pred, test_label):

    fpr_test, tpr_test, auc_test = get_plot(pred_list=test_pred, label_list=test_label)
    plt.plot(fpr_test, tpr_test, linewidth=3)  # 画出test
    plt.legend(["test (AUC = %0.3f)" % auc_test], loc="lower right")
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel('False Positive Rate', fontsize = 15)
    plt.ylabel('True Positive Rate', fontsize = 15)
    plt.title(' ROC curve', fontsize = 15)
    plt.show()

data_path = r"D:\data\result(3).csv"
pd_data = pd.read_csv(data_path)

row_case_name = pd_data["casename"]
row_prob = pd_data["prob"]

case_name = list(set([x.split("_")[-2] for x in row_case_name]))

big_label = []
mean_prob = []
max_prob = []

for case in case_name:
    label = []
    prob_list = []
    for c, p in zip(row_case_name, row_prob):
        if case == c.split("_")[2]:
            prob_list.append(p)
            label.append(c.split("_")[-1])
    # 均值预测
    mean_prob.append(np.mean(np.array(prob_list)))
    big_label.append(int(label[0]))
    max_prob.append(np.max(np.array(prob_list)))

max_prob = np.array(max_prob)
mean_prob = np.array(mean_prob)
big_label = np.array(big_label)
show_roc(test_pred=max_prob, test_label=big_label)
show_roc(test_pred=mean_prob, test_label=big_label)
for m, me, l in zip(max_prob, mean_prob, big_label):
    print(round(m,2), round(me, 2), l)



