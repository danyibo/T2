import numpy as np
import matplotlib.pyplot as plt
from MeDIT.Visualization import Imshow3DArray
import SimpleITK as sitk
import math
import cv2
import seaborn as sns
import pandas as pd
import os
import shutil


class CheckData:
    def __init__(self):
        self.data_folder_path = r'E:\split_data_ACLR_T2\ZHANG YU CHENG-20190601\store'
        self.store_folder_path = os.path.join(r'E:\split_data_ACLR_T2\ZHANG YU CHENG-20190601', "3D")
        if not os.path.exists(self.store_folder_path):
            os.makedirs(self.store_folder_path)
        self.roi_path = r"E:\split_data_ACLR_T2\ZHANG YU CHENG-20190601\roi.nii.gz"
        roi = sitk.ReadImage(self.roi_path)
        self.roi_array = sitk.GetArrayFromImage(roi)
        self.roi_array = np.where(self.roi_array == 2, 1, 0)

    def show(self):
        not_t2_data = r"E:\split_data_ACLR_T2\ZHANG YU CHENG-20190601\data_1.nii"
        data = sitk.ReadImage(not_t2_data)
        data_array = sitk.GetArrayFromImage(data)
        data_array = np.transpose(data_array, (1, 2, 0))
        print(self.roi_array.shape)
        print(data_array.shape)
        roi = np.transpose(self.roi_array, (1, 2, 0))
        Imshow3DArray(data_array, roi)

    def sort_image(self, image_path):
        list_new_image = []
        list_number = []
        image_list = os.listdir(image_path)
        for i in image_list:
            list_number.append(int(i.split("_")[-1]))
            list_number.sort()
        for j in list_number:

            for k in os.listdir(image_path):
                if j == int(k.split("_")[-1]):
                    list_new_image.append(k)
        return list_new_image

    def sort_file(self, dicom_folder_path):
        list_data = []
        new_dicom_folder_list = []
        dicom_folder_list = os.listdir(dicom_folder_path)
        for i in dicom_folder_list:
            list_data.append(int(i.split("_")[3].split(".")[0]))
            list_data.sort()
        for j in list_data:
            for k in os.listdir(dicom_folder_path):
                if j == int(k.split("_")[3].split(".")[0]):
                    new_dicom_folder_list.append(k)
        return new_dicom_folder_list

    def get_2D_data(self):
        data_path_list = self.sort_image(self.data_folder_path)
        for folder in data_path_list:
            folder_path = os.path.join(self.data_folder_path, folder)
            for file in os.listdir(folder_path):
                if file.split(".")[-1] == "npy":
                    image_path = os.path.join(folder_path, file)
                    shutil.copy(image_path, self.store_folder_path)

    def get_3D_array(self):
        T_2_array = []
        file_list = self.sort_file(self.store_folder_path)
        for file in reversed(file_list):
            file_path = os.path.join(self.store_folder_path, file)
            image_2D_array = np.load(file_path)
            T_2_array.append(image_2D_array)

        T_2_array = np.reshape(T_2_array, self.roi_array.shape)
        roi = np.transpose(self.roi_array, (1, 2, 0))
        T_2_array = np.transpose(T_2_array, (1, 2, 0))
        Imshow3DArray(T_2_array, roi)
        # for i in range(20):
        T_2_array_2D = T_2_array[:, :, 19]
        roi_array_2D = roi[:, :, 19]
        ax = sns.heatmap(T_2_array_2D)
        plt.show()

        mean_roi = []
        for i in np.reshape(T_2_array_2D * roi_array_2D, (self.roi_array.shape[1] * self.roi_array.shape[2])):
            if i != 0:
                mean_roi.append(i)
        print(np.mean(mean_roi))
        plt.hist(mean_roi, bins=40)
        plt.show()


check_data = CheckData()
check_data.get_2D_data()
check_data.get_3D_array()
check_data.show()
