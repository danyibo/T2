import os
import numpy as np
from Tool.DataProcress import get_array_from_path, standard
from MeDIT.Visualization import Imshow3DArray
import matplotlib.pyplot as plt
from Tool.FolderProcress import make_folder
from MeDIT.SaveAndLoad import SaveNumpyToImageByRef
import SimpleITK as sitk


class Check:
    """

    检查计算后的图像，将5个回波的应该删除

    """
    def __init__(self):
        self.root_path = r"X:\DYB\data_and_result\doctor_xie\knee_data"  # 存放膝盖数据的文件夹

    def sorted_file(self, folder_path):
        new_list = [int(i.split("_")[-1].split(".")[0]) for i in os.listdir(folder_path)]
        new_list.sort()  # 按照后面是序号进行排序

        sorted_folder = []
        for index in new_list:
            for file in os.listdir(folder_path):
                if file.split("_")[-1].split(".")[0] == str(index):
                    sorted_folder.append(file)
        return sorted_folder

    def get_image(self):
        """
        该函数用于：存放计算的三维T2 四回波
        :return:
        """
        for case in os.listdir(self.root_path):
            store_data_path = os.path.join(self.root_path, case, "T2_array")
            new_store_data = self.sorted_file(folder_path=store_data_path)



            roi_path = os.path.join(self.root_path, case, "xia_gu_down_1.nii")  # 原始ROI
            roi_array = get_array_from_path(roi_path)  # 将ROI的形状作为参考，以及保存为nii的参考图像
            ref_roi = sitk.ReadImage(roi_path)

            T_2_array = []  # 三维
            for file in reversed(new_store_data):  # 这里进行反转的原因感觉是因为reshape方法的问题
                file_path = os.path.join(self.root_path, case, "T2_array", file)
                image_2D = np.load(file_path)
                T_2_array.append(image_2D)
            change_roi = np.transpose(roi_array, (2, 1, 0))
            T_2_array = np.reshape(T_2_array, change_roi.shape)
            T_2_array = np.transpose(T_2_array, (1, 2, 0))
            import cv2
            T_2_array_filtered = cv2.blur(T_2_array, (5,5))
            T_2_array_filtered = np.asarray(T_2_array_filtered)
            # print(T_2_array)
            # SaveNumpyToImageByRef(os.path.join(self.root_path, case, "T2_filtered.nii"), T_2_array_filtered, ref_roi)
            SaveNumpyToImageByRef(os.path.join(self.root_path, case, "T2.nii"), T_2_array, ref_roi)

            print("case {} is saved".format(case))

    def get_T2_mean(self, array):
        """

        :param array: 传入T2和roi相乘的矩阵图
        :return: 返回均值
        """
        array = array.flatten()
        array = [i for i in array if i != 0]
        return sum(array) / len(array)

    def get_T2_value(self):

        T2_up_list = []  # 存放T2值
        T2_down_list = []
        for case in os.listdir(self.root_path):
            roi_array = get_array_from_path(os.path.join(self.root_path, case, "roi.nii.gz"))
            roi_array_1 = np.where(roi_array == 1, 1, 0)  # up
            roi_array_2 = np.where(roi_array == 2, 1, 0)  # up
            roi_array_3 = np.where(roi_array == 3, 1, 0)  # down
            roi_array_4 = np.where(roi_array == 4, 1, 0)  # down

            T2_path = os.path.join(self.root_path, case, "four_T2_array", "T2_3D_array.npy")



            T_2_array = np.load(T2_path)

            # T_2_array = standard(T_2_array)  # 这句只有显示的时候能打开,计算T2时，不能进行归一化
            roi_up = roi_array_1 + roi_array_2
            roi_down = roi_array_3 + roi_array_4

            np.save(os.path.join(self.root_path, case, "T2_up.npy"), T_2_array * roi_up)
            np.save(os.path.join(self.root_path, case, "T2_down.npy"), T_2_array * roi_down)

            T2_up_vaule = self.get_T2_mean(T_2_array * roi_up)
            T2_down_vaule = self.get_T2_mean(T_2_array * roi_down)
            T2_up_list.append(T2_up_vaule)
            T2_down_list.append(T2_down_vaule)

        result = {"CaseName": os.listdir(self.root_path),
                  "T2_up_vaule": T2_up_list,
                  "T2_down_value": T2_down_list}

        import pandas as pd
        pd_result = pd.DataFrame(result)
        store_path = os.path.dirname(self.root_path)
        pd_result.to_csv(os.path.join(store_path, "T2_value.csv"),index=None)


if __name__ == '__main__':
    check = Check()
    check.get_image()  # 将T2 二维的合成三维
    # check.get_T2_value()