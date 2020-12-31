import os
import numpy as np
import matplotlib.pyplot as plt
from Tool.DataProcress import get_array_from_path, standard
from MeDIT.Visualization import Imshow3DArray
import seaborn as sns
import cv2


class ShowData:
    def __init__(self):
        self.root_path = r"Y:\DYB\data_and_result\doctor_xie\knee_data"

    def get_single_roi(self, roi_array):
        """将多个拆为单个显示"""
        roi_array_1 = np.where(roi_array == 1, 1, 0)
        roi_array_2 = np.where(roi_array == 2, 1, 0)
        roi_array_3 = np.where(roi_array == 3, 1, 0)
        roi_array_4 = np.where(roi_array == 4, 1, 0)
        return [roi_array_1, roi_array_2, roi_array_3, roi_array_4]

    def show_origin_roi(self, value):
        """展示最原始的ROI，并去掉半月板"""
        for case in os.listdir(self.root_path):
            data_path = os.path.join(self.root_path, case, "new_data.nii")
            data_array = standard(get_array_from_path(data_path))
            roi_path = os.path.join(self.root_path, case, "roi.nii.gz")

            """展示原始ROI"""
            roi_array = get_array_from_path(roi_path)
            roi_list = self.get_single_roi(roi_array)
            # Imshow3DArray(data_array, roi_list)

            """展示软骨和下骨ROI"""
            bone_roi_path = os.path.join(self.root_path, case, "moved_new_roi.nii.gz") # 整数
            bone_roi_array = get_array_from_path(bone_roi_path)
            bone_roi_list = self.get_single_roi(bone_roi_array)

            Imshow3DArray(data_array, bone_roi_list[value])
            Imshow3DArray(data_array, roi_list[value])
            print(np.unique(bone_roi_array))

            """展示计算图像"""
            four_array = os.path.join(self.root_path, case, "four_T2_array", "T2_3D_array.npy")
            four_array = np.load(four_array)

            img_mean = cv2.blur(four_array, (5, 5))



            # for i in range(four_array.shape[-1]):
            #     sns.set()
            #     ax = sns.heatmap(img_mean[...,i])
            #     plt.show()
            break

    # def remove(self):
    #     """删除没用的文件"""
    #     for case in os.listdir(self.root_path):
    #         T2_path = os.path.join(self.root_path, case, "T2.nii")
    #         os.remove(T2_path)
    #         T2_array = os.path.join(self.root_path, case, "3D_T2_array", "T2_3D_array.npy")
    #         os.remove(T2_array)
    #         folder = os.path.join(self.root_path, case, "3D_T2_array")
    #         os.rmdir(folder)






if __name__ == '__main__':
    show_data = ShowData()
    show_data.show_origin_roi(value=4)
    # show_data.remove()