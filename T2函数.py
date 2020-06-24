import os
import SimpleITK as sitk
import pydicom
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pydicom
import math
import pandas as pd
import time
from tqdm import tqdm
from scipy import signal


def comput_t2_by_five_point(p_1, p_2, p_3, p_4, p_5, TE_1, TE_2, TE_3, TE_4, TE_5):
    TE = np.array([TE_1, TE_2, TE_3, TE_4, TE_5]).reshape(-1, 1)
    ln_S_list = []  # 信号强度取对数
    S_list = [p_1, p_2, p_3, p_4, p_5]  # 原始信号强度
    for i in S_list:
        if i == 0:
            ln_S_list.append(i)  # 为零，则取对数也设置为零
        else:
            ln_S_list.append(math.log(i))
    model = LinearRegression()  # 拟合直线模型
    model.fit(TE, ln_S_list)
    if model.coef_ == 0:
        T2 = 0
        return T2
    if abs(model.coef_) < 0.001:  # 如果斜率接近于零，则T2值为零，否则出现很大值
        T2 = 0
        return T2
    else:
        T2 = -1 / model.coef_  # 由斜率求出T2
        return int(T2)

