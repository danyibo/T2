import os
import numpy as np
from Tool.DataProcress import get_array_from_path, standard
from MeDIT.Visualization import Imshow3DArray
import cv2
from MeDIT.SaveAndLoad import SaveArrayToNiiByRef
import SimpleITK as sitk
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from Tool.FolderProcress import make_folder


#########################
# 展示T2的图像
#########################

def show_T2(root_path):
    """查看一下图像"""
    for case in os.listdir(root_path):
        T2_array = get_array_from_path(os.path.join(root_path, case, "T2.nii"))
        roi_array = get_array_from_path(os.path.join(root_path, case, "ruan_gu_up_1.nii"))
        T2_array = standard(T2_array)
        Imshow3DArray(T2_array, roi_array)


def get_new_t2(root_path):
    """对图像进行平滑"""
    for case in os.listdir(root_path):
        t2_path = os.path.join(root_path, case, "T2.nii")
        t2_array = get_array_from_path(t2_path)
        # roi_array = get_array_from_path(os.path.join(root_path, case, "ruan_gu_up_1.nii"))
        new_t2_array = cv2.blur(t2_array, (5, 5))
        new_t2_array = new_t2_array.astype(np.int64)

        t2 = sitk.ReadImage(t2_path)
        SaveArrayToNiiByRef(store_path=os.path.join(root_path, case, "new_T2.nii"),
                            array=new_t2_array,
                            ref_image=t2)
        print("Case {} is finished!".format(case))


# root_path = r"Y:\DYB\data_and_result\doctor_xie\knee_data"
# get_new_t2(root_path)

##############################
# 计算相关性系数  2020年10月12日
##############################

class GetCorrelation:
    def __init__(self, feature_name, t2_name):
        self.folder_path = r"Y:\DYB\data_and_result\doctor_xie\Feature"
        self.t2_path = r"Y:\DYB\data_and_result\doctor_xie\Feature\T2_value.csv"
        self.feature_name = feature_name
        self.t2_name = t2_name
        self.feature_path = os.path.join(self.folder_path, feature_name+".csv")
        self.pd_t2 = pd.read_csv(self.t2_path)
        self.pd_feature = pd.read_csv(self.feature_path)
        self.t2_value = self.pd_t2[t2_name]

    def remove_bad_data(self, t2_value, feature_value):
        new_t2_value = []
        new_feature_value = []
        t2_min = sorted(t2_value)[3] # 去除最小的点
        # t2_max = sorted(t2_value, reverse=True)[0]  # 去除一个最大的点
        # print(t2_max)
        for t, f in zip(t2_value, feature_value):
            if t > t2_min:
                new_t2_value.append(t)
                new_feature_value.append(f)
        return new_t2_value, new_feature_value

    def get_r_p_feature_name(self, sorted_number, show=False):
        r_list = []
        p_list = []

        for feature_name in self.pd_feature.columns[1:]:  # 注意不要casename
            feature_value = self.pd_feature[feature_name]
            new_t2_value, new_feature_value = self.remove_bad_data(t2_value=self.t2_value, feature_value=feature_value)
            r, p = pearsonr(new_t2_value, new_feature_value)
            if p < 0.05:
                r_list.append(r); p_list.append(p)
        r_min = sorted([abs(i) for i in r_list], reverse=True)[sorted_number-1]

        new_r_list = []
        new_p_list = []
        new_feature_name = []
        for feature_name in self.pd_feature.columns[1:]:  # 注意不要casename
            feature_value = self.pd_feature[feature_name]
            new_t2_value, new_feature_value = self.remove_bad_data(t2_value=self.t2_value, feature_value=feature_value)
            r, p = pearsonr(new_t2_value, new_feature_value)
            if p < 0.05 and abs(r) >= r_min:  # 一定要注意不要忘记绝对值
                r = round(r, 4)
                p = round(p, 4)
                new_r_list.append(r)
                new_p_list.append(p)
                new_feature_name.append(feature_name)
                ####################
                # 显示相关系数图
                ####################
                if show is True:
                    sns.regplot(x=new_feature_value, # 特征值
                                y=new_t2_value,  # t2值

                                scatter_kws={'s': 15},
                                )
                    plt.xlabel(feature_name)
                    plt.ylabel("T2 value")
                    plt.title("correlation value = " + str(round(r, 4)))
                    store_path = make_folder(os.path.join(self.folder_path, self.feature_name))
                    plt.savefig(os.path.join(store_path, feature_name+".png"))
                    plt.clf()

                    # plt.show()
                #####################
        p_r_feature_name_result = pd.DataFrame(data={"feature_name":new_feature_name,
                  "r_value":new_r_list,
                  "p_value":new_p_list})
        store_path = make_folder(os.path.join(self.folder_path, self.feature_name))
        p_r_feature_name_result.to_csv(os.path.join(store_path, "correlation_index.csv"), index=None)


if __name__ == '__main__':
    # get_result = GetCorrelation(feature_name="xia_gu_up_1", t2_name="mean_up_1")
    # get_result.get_r_p_feature_name(5, show=True)
    # get_result = GetCorrelation(feature_name="xia_gu_up_2", t2_name="mean_up_2")
    # get_result.get_r_p_feature_name(5, show=True)
    # get_result = GetCorrelation(feature_name="xia_gu_down_1", t2_name="mean_down_1")
    # get_result.get_r_p_feature_name(5, show=True)
    get_result = GetCorrelation(feature_name="xia_gu_down_2", t2_name="mean_down_2")
    get_result.get_r_p_feature_name(5, show=True)





