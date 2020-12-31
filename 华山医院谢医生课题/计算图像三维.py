"""
直接从三维的图像进行拟合计算出T2值
"""

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


class GetT2Mapping:
    def __init__(self):
        self.case_path = r"Y:\DYB\2020832DATA\doctor_xie\normal_control\case0\nii_folder"
        self.nii_file_list = os.listdir(self.case_path)

    def get_t2(self):
        nii_file_list = [os.path.join(self.case_path, i) for i in self.nii_file_list]
        print(nii_file_list)


get_t2_mapping = GetT2Mapping()
get_t2_mapping.get_t2()