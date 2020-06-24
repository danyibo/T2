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


plt.style.use("ggplot")


class ComputT2:
    def __init__(self, dicom_folder_path):
        # self.dicom_folder_path = r"E:\split_data_ACLR_T2\RUAN XIAO BAI-20140430"
        # self.sotre_case_path = r"E:\split_data_ACLR_T2\RUAN XIAO BAI-20140430\store"
        self.dicom_folder_path = dicom_folder_path
        self.sotre_case_path = os.path.join(dicom_folder_path, "store")



    def get_array(self, dicom_file_path):
        image = sitk.ReadImage(dicom_file_path)
        image_array = sitk.GetArrayFromImage(image)
        image_shape = image_array.shape
        image_array = np.reshape(image_array, (image_array.shape[0] * image_array.shape[1] * image_array.shape[2]))
        image_info = pydicom.dcmread(dicom_file_path)
        TE = image_info.EchoTime
        return image_array, TE, image_shape

    def sort_file(self, dicom_folder_path):
        list_data = []
        new_dicom_folder_list = []
        dicom_folder_list = os.listdir(dicom_folder_path)
        for i in dicom_folder_list:
            if i.split(".")[-1] == "dcm":
                list_data.append(int(i.split("-")[1].split(".")[0]))
                list_data.sort()
        for j in list_data:
            for k in os.listdir(dicom_folder_path):
                if k.split(".")[-1] == "dcm":
                    if j == int(k.split("-")[1].split(".")[0]):
                        new_dicom_folder_list.append(k)
        return new_dicom_folder_list

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

    def comput_t2_by_five_point(self, p_1, p_2, p_3, p_4, p_5, TE_1, TE_2, TE_3, TE_4, TE_5):
        TE = np.array([TE_1, TE_2, TE_3, TE_4, TE_5]).reshape(-1, 1)
        ln_S_list = []  # 信号强度取对数
        S_list = [p_1, p_2, p_3, p_4, p_5]  # 原始信号强度
        for i in S_list:
            if i == 0:
                ln_S_list.append(i)   # 为零，则取对数也设置为零
            else:
                ln_S_list.append(math.log(i))
        model = LinearRegression()  # 拟合直线模型
        model.fit(TE, ln_S_list)
        # y_plot = model.predict(TE)
        # plt.scatter(TE, ln_S_list, color='red', label="point", linewidth=7, alpha=0.8)
        # plt.plot(TE, y_plot, color="green", label="line", linewidth=4, alpha=0.8)
        # # plt.xlabel("TE")
        # # plt.ylabel("S")
        # # plt.legend()
        # plt.show()
        if model.coef_ == 0:
            T2 = 0
            return T2
        if abs(model.coef_) < 0.01:  # 如果斜率接近于零，则T2值为零，否则出现很大值
            T2 = 0
            return T2
        else:
            T2 = -1/model.coef_  # 由斜率求出T2
            return int(T2)

    def Get_T2(self):
        dicom_folder_1 = os.path.join(self.dicom_folder_path, "folder_1")
        dicom_folder_2 = os.path.join(self.dicom_folder_path, "folder_2")
        dicom_folder_3 = os.path.join(self.dicom_folder_path, "folder_3")
        dicom_folder_4 = os.path.join(self.dicom_folder_path, "folder_4")
        dicom_folder_5 = os.path.join(self.dicom_folder_path, "folder_5")

        dicom_folder_1_list = self.sort_file(dicom_folder_1)
        dicom_folder_2_list = self.sort_file(dicom_folder_2)
        dicom_folder_3_list = self.sort_file(dicom_folder_3)
        dicom_folder_4_list = self.sort_file(dicom_folder_4)
        dicom_folder_5_list = self.sort_file(dicom_folder_5)

        all_length = len(dicom_folder_1_list) + len(dicom_folder_2_list) + \
                     len(dicom_folder_3_list) + len(dicom_folder_4_list) + len(dicom_folder_5_list)
        index_number = int(all_length / 5)
        for i in range(0, index_number):
            image_path_1 = os.path.join(dicom_folder_1, dicom_folder_1_list[i])
            image_path_2 = os.path.join(dicom_folder_2, dicom_folder_2_list[i])
            image_path_3 = os.path.join(dicom_folder_3, dicom_folder_3_list[i])
            image_path_4 = os.path.join(dicom_folder_4, dicom_folder_4_list[i])
            image_path_5 = os.path.join(dicom_folder_5, dicom_folder_5_list[i])
            store_image_path = os.path.join(self.sotre_case_path, "image_" + str(i))
            if not os.path.exists(store_image_path):
                os.makedirs(store_image_path)
            shutil.copy(image_path_1, store_image_path)
            shutil.copy(image_path_2, store_image_path)
            shutil.copy(image_path_3, store_image_path)
            shutil.copy(image_path_4, store_image_path)
            shutil.copy(image_path_5, store_image_path)

    def get_3D_image(self):

        image_list = self.sort_image(self.sotre_case_path)

        for folder in image_list:
            T_2_list = []
            folder_image_path = os.path.join(self.sotre_case_path, folder)
            dicom_name = self.sort_file(folder_image_path)


            dicom_path_1 = os.path.join(folder_image_path, dicom_name[0])
            dicom_path_2 = os.path.join(folder_image_path, dicom_name[1])
            dicom_path_3 = os.path.join(folder_image_path, dicom_name[2])
            dicom_path_4 = os.path.join(folder_image_path, dicom_name[3])
            dicom_path_5 = os.path.join(folder_image_path, dicom_name[4])
            dicom_file_1_array, TE_1, image_shape = self.get_array(dicom_path_1)
            dicom_file_2_array, TE_2, image_shape = self.get_array(dicom_path_2)
            dicom_file_3_array, TE_3, image_shape = self.get_array(dicom_path_3)
            dicom_file_4_array, TE_4, image_shape = self.get_array(dicom_path_4)
            dicom_file_5_array, TE_5, image_shape = self.get_array(dicom_path_5)


            for p_1, p_2, p_3, p_4, p_5 in zip(dicom_file_1_array,
                                               dicom_file_2_array,
                                               dicom_file_3_array,
                                               dicom_file_4_array,
                                               dicom_file_5_array,):
                # print(p_1, os.path.basename(dicom_path_1), p_2, os.path.basename(dicom_path_2), p_3, os.path.basename(dicom_path_3), p_4, os.path.basename(dicom_path_4), p_5, os.path.basename(dicom_path_5))

                T2 = self.comput_t2_by_five_point(p_1, p_2, p_3, p_4, p_5, TE_1, TE_2, TE_3, TE_4, TE_5)
                print(T2)
                T_2_list.append(T2)

            T_2_array = np.array(T_2_list)
            T_2_array = np.reshape(T_2_array, image_shape)
            # T_2_array = signal.medfilt(T_2_array, (3, 3))

            np.save(os.path.join(folder_image_path, "T_2_array" + folder + ".npy"), T_2_array)


for case in os.listdir(r'E:\split_data_ACLR_T2'):
    case_path = os.path.join(r'E:\split_data_ACLR_T2', case)
    comput_T2 = ComputT2(dicom_folder_path=case_path)
    comput_T2.Get_T2()
    comput_T2.get_3D_image()
