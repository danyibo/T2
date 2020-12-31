import os
import numpy as np
import pandas as pd
from Tool.DataProcress import get_array_from_path, standard
from MeDIT.Visualization import Imshow3DArray

import matplotlib.pyplot as plt
import seaborn as sns
# plt.style.use("ggplot")

class ShowKnne:
    def __init__(self):
        self.case_path = r"Y:\DYB\data_and_result\doctor_xie\knee_data\CAI NAI JI-20140605"

    def show_array(self, array):
        plt.imshow(array[..., 5], cmap="gray")
        plt.axis('off')
        # plt.show()

    def show_t2(self, array):
        plt.imshow(array[..., 5], cmap="gray")
        plt.axis('off')
        # plt.show()

    def show_knee_array(self):
        five_echo_name_list = ["data_" + str(i) for i in range(1, 6)]
        five_echo_path_list = [os.path.join(self.case_path, i) for i in five_echo_name_list]
        five_array_list = [np.flipud(get_array_from_path(i)) for i in five_echo_path_list]
        # # 显示五个回波的图像
        for echo, i in zip(five_array_list[1:], range(1, 6)):
            plt.subplot(4, 1, i)
            self.show_array(echo)
        plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
        plt.show()

    def show_T2_mampping(self):
        t2_mapping_path = os.path.join(self.case_path, "new_T2.nii")
        t2_mapping_array = get_array_from_path(t2_mapping_path)
        # sns.set()
        # sns.heatmap(t2_mapping_array[..., 5])
        # plt.axis('off')
        # plt.show()

        ruan_roi_path = r"Y:\DYB\data_and_result\doctor_xie\knee_data\CAI NAI JI-20140605\ruan_gu_down_1.nii"
        xia_roi_path = r"Y:\DYB\data_and_result\doctor_xie\knee_data\CAI NAI JI-20140605\xia_gu_down_1.nii"
        ruan_array = get_array_from_path(ruan_roi_path)
        xia_array = get_array_from_path(xia_roi_path)
        for i in range(t2_mapping_array.shape[-1]):
            if np.sum(ruan_array[..., i]) != 0 and np.sum(xia_array[..., i]) != 0:
                sns.heatmap(t2_mapping_array[..., i])
                plt.contour(ruan_array[..., i], colors="blue", linewidths=0.45)
                plt.contour(xia_array[..., i], linewidths=0.45)
                plt.axis('off')
                plt.show()


show_knne = ShowKnne()
show_knne.show_knee_array()
show_knne.show_T2_mampping()