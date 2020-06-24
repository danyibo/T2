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

plt.style.use("ggplot")




class CheckData:
    def __init__(self, data_folder_path, store_folder_path, roi_path):
        self.data_folder_path = data_folder_path
        self.store_folder_path = store_folder_path
        self.roi_path = roi_path
        roi = sitk.ReadImage(self.roi_path)
        self.roi_array = np.transpose(sitk.GetArrayFromImage(roi), (1, 2, 0))
        self.roi_array_1 = np.where(self.roi_array == 1, 1, 0)
        self.roi_array_2 = np.where(self.roi_array == 2, 1, 0)
        self.roi_array_3 = np.where(self.roi_array == 3, 1, 0)
        self.roi_array_4 = np.where(self.roi_array == 4, 1, 0)  # 下面那个roi
        self.roi_array_5 = np.where(self.roi_array == 5, 1, 0)
        self.roi_array_6 = np.where(self.roi_array == 6, 1, 0)
        self.roi_array = [self.roi_array_1,self.roi_array_2,self.roi_array_3,self.roi_array_4,self.roi_array_5,self.roi_array_6]
        # self.roi_array = [self.roi_array_4,]
        # self.roi_array = np.where(self.roi_array == 2, 1, 0)
        # self.roi_array = np.where(self.roi_array == 4, 1, 0)

    def stand(self, data):
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        return data

    def move_roi(self, roi_array):
        roi_shape = roi_array.shape
        moved_value = 12
        roi_big = roi_array[:roi_shape[0]-moved_value, :, :]
        roi_small = roi_array[roi_shape[0]-moved_value:, :, :]
        new_roi_array = np.vstack((roi_small, roi_big))
        return new_roi_array

    def show(self):
        not_t2_data = r"E:\split_data_ACLR_T2\RUAN XIAO BAI-20140430\data_1.nii"
        data = sitk.ReadImage(not_t2_data)
        data_array = sitk.GetArrayFromImage(data)
        data_array = np.transpose(data_array, (1, 2, 0))

        # roi = np.transpose(self.roi_array, (1, 2, 0))
        data_array = self.stand(data_array)
        data_array = np.flipud(data_array)
        Imshow3DArray(data_array, self.roi_array)

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

        change_roi = np.transpose(self.roi_array_1, (2, 1, 0))  # 将数据的形状reshape到roi大小

        T_2_array = np.reshape(T_2_array, change_roi.shape)
        T_2_array = np.transpose(T_2_array, (1, 2, 0))



        # T_2_array = self.stand(T_2_array)
        # Imshow3DArray(T_2_array, self.roi_array)
        return T_2_array


    def get_roi_4_T2(self):
        T2 = []
        image_T2_array = self.get_3D_array()
        T2_array = image_T2_array * self.roi_array_4
        T2_array_flatten = T2_array.flatten()
        for value in T2_array_flatten:
            if value != 0:
                T2.append(value)
        T2_mean_value = sum(T2) / len(T2)
        return T2_mean_value

    def get_moved_roi_T2(self):
        T2_down = []
        moved_roi = self.move_roi(self.roi_array_4)
        image_array = self.get_3D_array()
        # image_array = self.stand(image_array)
        # Imshow3DArray(image_array, moved_roi)
        T2_down_array = image_array * moved_roi
        T2_down_array = T2_down_array.flatten()
        for value in T2_down_array:
            if value != 0:
                T2_down.append(value)
        T2_down_mean = sum(T2_down) / len(T2_down)
        return T2_down_mean




def median(data):
    data.sort()
    half = len(data) // 2
    return (data[half] + data[~half])/2


def make_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)





def chech_all_data(all_data_path):
    T_2_list = []
    T_2_down = []
    case_name = []
    for case in os.listdir(all_data_path):
        case_path = os.path.join(all_data_path, case)
        store_3D_path = os.path.join(case_path, "3D")
        make_folder(store_3D_path)

        data_folder_path = os.path.join(case_path, "store")
        roi_path = os.path.join(case_path, "roi.nii.gz")

        check_data = CheckData(data_folder_path=data_folder_path, store_folder_path=store_3D_path, roi_path=roi_path)
        check_data.get_2D_data()
        check_data.get_3D_array()
        T2 = check_data.get_roi_4_T2()
        # T2_dowm = check_data.get_moved_roi_T2()
        # T_2_down.append(T2_dowm)
        T_2_list.append(T2)
        # print("软骨下骨均值",T2_dowm)
        print("软骨均值", T2)

        case_name.append(case)

    plt.hist(T_2_list, edgecolor='black')
    plt.plot(x=36.33)
    plt.title("T2 mean value")
    plt.show()

    # result = pd.DataFrame({"Case":case_name,
    #                        "T2_value":T_2_list}, index=None)
    # result.to_csv(r'E:\膝盖T2值\T2_case_label_518.csv')
    #
    # result_2 = pd.DataFrame({"Case": case_name,
    #                        "T2_value": T_2_down}, index=None)
    # result_2.to_csv(r'E:\膝盖T2值\T2_down_case_label.csv')



all_data_path = r"E:\split_data_ACLR_T2"
chech_all_data(all_data_path)