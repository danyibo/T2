import os
import pandas as pd
import numpy as np

from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

data_path = r"E:\knn_feature_5_21\T2_all_feature.csv"

pd_data = pd.read_csv(data_path)

p_value = []
feature_list = []
r_list = []

T2 = pd_data["T2_value"]
feature_name = list(pd_data.columns)
for feaure in feature_name[2:]:
    feature_array = pd_data[feaure]
    r, p = pearsonr(T2, feature_array)
    r = round(r, 3)
    p = round(p, 3)
    if p < 0.05 and abs(r) > 0.45:
        print("相关系数r: {}, p_value: {}, feature_name: {}".format(r, p, feaure))
        r_list.append(r)
        p_value.append(p)
print(np.max(r_list))
print(len(r_list))
    #     feature_list.append(feaure)

# result_2 = pd.DataFrame({"feature": feature_list, "p_value": p_value, "r": r_list}, index=None)
# result_2.to_csv(r'E:\knn_feature\p_feature_518.csv')
