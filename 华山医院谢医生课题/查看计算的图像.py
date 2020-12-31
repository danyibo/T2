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
import sklearn
from Tool.DataProcress import get_array_from_path, standard
from Tool.FolderProcress import make_folder
from Tool.Visualization import show_array

# plt.style.use("ggplot")


class CheckData:
    def __init__(self, data_folder_path, store_folder_path):
        self.data_folder_path = data_folder_path
        self.store_folder_path = make_folder(store_folder_path)

    def stand(self, data):
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        return data

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
            list_data.append(int(i.split("_")[-1].split(".")[0]))
        list_data.sort(reverse=False)
        for j in list_data:
            for k in os.listdir(dicom_folder_path):
                if j == int(k.split("_")[-1].split(".")[0]):
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
        # T_2_array = []
        # file_list = self.sort_file(self.store_folder_path)
        # for file in file_list:
        #     file_path = os.path.join(self.store_folder_path, file)
        #     image_2D_array = np.load(file_path)
        #     T_2_array.append(image_2D_array[0, ...])
        #     plt.imshow(image_2D_array[0, ...])
        #     plt.show()

        # T_2_array = np.array(T_2_array)
        # T_2_array = np.transpose(T_2_array, (1, 2, 0))
        # print(T_2_array.shape)
        data_path = r"Y:\DYB\2020832DATA\doctor_xie\normal_control\case0\nii_folder\T2MAPPING_anatomical_LEFT_e1.nii"
        data_array = get_array_from_path(data_path)
        data_array = np.flipud(data_array)

        Imshow3DArray(standard(data_array))
        # Imshow3DArray(standard(T_2_array))


if __name__ == '__main__':
    root_path = r"Y:\DYB\2020832DATA\doctor_xie\normal_control"
    for case in os.listdir(root_path):
        case_path = os.path.join(root_path, case)
        data_folder_path = os.path.join(case_path, "store_dicom")
        store_folder_path = os.path.join(case_path, "3D")
        check_data = CheckData(data_folder_path, store_folder_path)
        # 先将数据放在3D的文件夹中
        # check_data.get_2D_data()
        print("Case {} is finished!".format(case))
        # 合成三维的图像
        check_data.get_3D_array()
        break