import os
import SimpleITK as sitk
import pydicom
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math
plt.style.use("ggplot")
from Tool.DataProcress import get_array_from_path
from Tool.FolderProcress import make_folder


class GetT2:
    def __init__(self):
        # 这里最后初始化的时候，就传入case_path就可以了
        self.case_path = r"Y:\DYB\2020832DATA\doctor_xie\normal_control\case0"

        self.nii_folder_path = os.path.join(self.case_path, "nii_folder")
        self.dcm_folder_path = os.path.join(self.case_path, "all_dicom")
        self.store_dicom_path = make_folder(os.path.join(self.case_path, "store_dicom"))

    def get_image_array_and_te(self, dicom_file_path):
        """dicom_file_path 是一张dicom的图 """
        image = sitk.ReadImage(dicom_file_path)
        image_array = sitk.GetArrayFromImage(image)
        image_shape = image_array.shape
        # image_array是将一张dicom图像压成一个向量
        image_array = np.reshape(image_array, (image_array.shape[0] * image_array.shape[1] * image_array.shape[2]))
        image_info = pydicom.dcmread(dicom_file_path)
        TE = image_info.EchoTime
        return image_array, TE, image_shape

    def split_dicom(self):
        """将五个回波的图像再进行一次拆分"""
        folder_name_list = ["folder_" + str(i) for i in range(1, 6)]
        folder_path_list = [os.path.join(self.case_path, f) for f in folder_name_list]






get_t2 = GetT2()
get_t2.split_dicom()



