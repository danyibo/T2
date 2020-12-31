import os
import numpy as np
import matplotlib.pyplot as plt
from Tool.DataProcress import get_array_from_path, standard
from MeDIT.Visualization import Imshow3DArray
import seaborn as sns
import cv2
import SimpleITK as sitk
from MeDIT.SaveAndLoad import SaveArrayToNiiByRef
import pandas as pd


"""注意数值"""


class GetRelation:
    def __init__(self, case_path):
        self.case_path = case_path

    def get_ROI(self, value, store_name): # value用来选择
        ruangu_roi = os.path.join(self.case_path, "roi.nii.gz")  # 软骨路径
        xiagu_roi = os.path.join(self.case_path, "moved_new_roi.nii.gz") # 下骨路径

        data_path = os.path.join(self.case_path, "data_1.nii")
        data_array = standard(np.flipud(get_array_from_path(data_path)))
        data = sitk.ReadImage(data_path)


        ruangu_roi_array = get_array_from_path(ruangu_roi)
        single_ruangu_array = np.where(ruangu_roi_array == value, 1, 0)  # 软骨


        xiagu_roi_array = get_array_from_path(xiagu_roi)
        single_xiagu_array = np.where(xiagu_roi_array == value, 1, 0)  # 下骨

        # 显示检查
        # Imshow3DArray(data_array, [single_ruangu_array, single_xiagu_array])
        SaveArrayToNiiByRef(os.path.join(self.case_path, "ruan_gu_"+store_name+".nii"), single_ruangu_array, data)
        SaveArrayToNiiByRef(os.path.join(self.case_path, "xia_gu_"+store_name+".nii"), single_xiagu_array, data)


def run(root_path):
    """该函数是为了得到软骨，软骨下骨部位独立的ROI"""
    for case in os.listdir(root_path):
        case_path = os.path.join(root_path, case)
        get_relation = GetRelation(case_path=case_path)
        get_relation.get_ROI(value=1, store_name="up_1") # 1 为上软骨
        get_relation.get_ROI(value=2, store_name="up_2") # 2 为上软骨
        get_relation.get_ROI(value=3, store_name="down_1") # 3 为下软骨
        get_relation.get_ROI(value=4, store_name="down_2") # 4 为下软骨
        print("Case {} is finished!".format(case))


"""检查T2图像和对应ROI"""
class Check:
    """检查"""
    def __init__(self, case_path):
        self.case_path = case_path

    def show_T2_and_ROI(self):
        T2_path = os.path.join(self.case_path, "T2.nii")
        roi_list = ["xia_gu_down_1.nii", "xia_gu_down_2.nii",
                    "xia_gu_up_1.nii", "xia_gu_up_2.nii"]
        T2_array = standard(get_array_from_path(T2_path))

        T2_array = cv2.blur(T2_array, (5, 5))
        for roi in roi_list:
            roi_path = os.path.join(self.case_path, roi)
            roi_array = get_array_from_path(roi_path)
            Imshow3DArray(T2_array, roi_array)

def run_check(root_path):
    """展示计算的T2图像和ROI"""
    for case in os.listdir(root_path):
        case_path = os.path.join(root_path, case)
        check_and_show = Check(case_path)
        check_and_show.show_T2_and_ROI()




class GetT2Value:
    def __init__(self, case_path): # roi_name 就是对应软骨的名称
        self.case_path = case_path
        self.T2_array_path = os.path.join(case_path, "T2.nii")
        self.T2_array = get_array_from_path(self.T2_array_path)

    def get_T2_mean(self, roi_name): # 对应软骨名称
        roi_array = get_array_from_path(os.path.join(self.case_path, roi_name+".nii"))
        old_array = self.T2_array * roi_array
        array = old_array.flatten()
        new_array = [i for i in array if i != 0]
        return sum(new_array) / len(new_array)

    def get_T2_ninety(self, roi_name):
        roi_array = get_array_from_path(os.path.join(self.case_path, roi_name+".nii"))
        old_array = self.T2_array * roi_array
        array = np.unique(old_array)
        ninety_value = np.percentile(array, 90)
        return ninety_value


def get_T2(root_path, store_path):
    """计算出T2均值和百分之九十分位置, 并存为表格"""

    T2_up_1_mean_list = []  # 上软骨1 均值
    T2_up_2_mean_list = []  # 上软骨2 均值

    T2_up_1_ninety_list = []  # 上软骨1 % 90
    T2_up_2_ninety_list = []  # 上软骨2 % 90

    T2_down_1_mean_list = []  # 下软骨1 均值
    T2_down_2_mean_list = []  # 下软骨2 均值

    T2_down_1_ninety_list = []  # 下软骨1 % 90
    T2_down_2_ninety_list = []  # 下软骨2 % 90

    for case in os.listdir(root_path):
        case_path = os.path.join(root_path, case)
        get_T2_value = GetT2Value(case_path=case_path)

        T2_up_1_mean_list.append(get_T2_value.get_T2_mean(roi_name="ruan_gu_up_1"))
        T2_up_2_mean_list.append(get_T2_value.get_T2_mean(roi_name="ruan_gu_up_2"))
        T2_up_1_ninety_list.append(get_T2_value.get_T2_ninety(roi_name="ruan_gu_up_1"))
        T2_up_2_ninety_list.append(get_T2_value.get_T2_ninety(roi_name="ruan_gu_up_2"))

        T2_down_1_mean_list.append(get_T2_value.get_T2_mean(roi_name="ruan_gu_down_1"))
        T2_down_2_mean_list.append(get_T2_value.get_T2_mean(roi_name="ruan_gu_down_2"))
        T2_down_1_ninety_list.append(get_T2_value.get_T2_ninety(roi_name="ruan_gu_down_1"))
        T2_down_2_ninety_list.append(get_T2_value.get_T2_ninety(roi_name="ruan_gu_down_2"))
        print(case)

    result = {"CaseName": os.listdir(root_path),
              "mean_up_1": T2_up_1_mean_list,
              "mean_up_2": T2_up_2_mean_list,
              "mean_down_1": T2_down_1_mean_list,
              "mean_down_2": T2_down_2_mean_list,

              "ninety_up_1": T2_up_1_ninety_list,
              "ninety_up_2": T2_up_2_ninety_list,
              "ninety_down_1": T2_down_1_ninety_list,
              "ninety_down_2": T2_down_2_ninety_list
              }
    pd_result = pd.DataFrame(result)
    pd_result.to_csv(os.path.join(store_path, "T2_value.cvs"), index=None)


from scipy.stats import pearsonr


class GetRValue:
    """计算得出R值和P值以及特征名字"""
    def __init__(self, feature_path, T2_value_list, T2_name):
        # feature_path 是提取的特征表格
        # T2_value_list 是对应的T2值列表
        # 务必注意特征和T2要一一对应
        self.feature_path = feature_path
        self.pd_feature = pd.read_csv(feature_path)
        self.T2_value_list = T2_value_list
        self.T2_name = T2_name


    """
    
       关于相关性的计算，应该先把所有的相关性先算出来，存成一个字典  key就是相关性，value是特征值和T2
       再把相关系数进行排序，显示相关系数最大的十个特征    
    
    """

    def remove_bad_points(self, T2_list, feature_list):
        """去掉计算相关性系数的时候的最小的五个点"""
        T2 = sorted(list(T2_list))
        min_T2_point = T2[5]  # 可以选择去掉几个点
        new_T2_list = []
        new_feature_list = []
        for t2, feature in zip(T2_list, feature_list):
            if t2 > min_T2_point:
                new_T2_list.append(t2)
                new_feature_list.append(feature)
        return new_T2_list, new_feature_list


    def get_main_R(self):
        """得到相关系数最大的前十个"""
        R_list = []
        feature_name = list(self.pd_feature.columns)[1:]  # 不包括casename
        for feature in feature_name:
            feature_column = self.pd_feature[feature]  # 得到特征的值
            T2_list, feature_value = self.remove_bad_points(T2_list=self.T2_value_list,
                                                            feature_list=feature_column)

            r, p = pearsonr(T2_list, feature_value)  # feature_column是特征的值
            if p <= 0.05:  # 如果满足统计检验
                R_list.append(abs(round(r, 4)))  # 将绝对值加入
        R_sorted = sorted(R_list, reverse=True)
        return R_sorted[5]



    def get_R_P_feature_name(self):
        """得到R, P， feature_name 并返回一个列表"""
        R_list = []
        P_list = []
        FeatureName = []

        R_min_10 = self.get_main_R()

        feature_name = list(self.pd_feature.columns)[1:]  # 不包括casename
        for feature in feature_name:
            feature_column = self.pd_feature[feature]  # 得到特征的值
            T2_list, feature_value = self.remove_bad_points(T2_list=self.T2_value_list,
                                                            feature_list=feature_column)

            r, p = pearsonr(T2_list, feature_value) # feature_column是特征的值
               # 得到前十个最大的R
            if p <= 0.05 and abs(r) > R_min_10:  # 如果满足统计检验
            # if p <= 0.05:
                print("R = {}, feature is {}".format(r, feature))
                P_list.append(round(p, 4))
                R_list.append(round(r, 4))
                FeatureName.append(feature)
                ######################################
                ##           拟合直线部分             ##
            #   ######################################
            #
            #     # """进行相关性拟合"""
            #     # """拟合R>0.5的直线"""
                sns.regplot(x=T2_list,  # T2值
                            y=feature_value,  # 特征值

                            scatter_kws={'s':15},
                            )
                plt.xlabel("T2_value ")
                plt.ylabel(feature)
                plt.title("correlation value = " + str(round(r, 4)))
                plt.show()
                ##################################################################################
                ##################################################################################

        result = {"feature_name":FeatureName,
                  "R value":R_list,
                  "P value":P_list}
        pd_result = pd.DataFrame(result)
        sub_store_path = os.path.dirname(self.feature_path)
        store_name = os.path.basename(self.feature_path)

        pd_result.to_csv(os.path.join(sub_store_path, "correlation_" + self.T2_name + "_" + store_name), index=None)


def get_R_P(feature_name, T2_name):
    """计算相关系数"""
    row_path = r"Y:\DYB\data_and_result\doctor_xie\Feature"

    feature_path = os.path.join(row_path, feature_name+".csv")
    T2_value_path = os.path.join(row_path, "T2_value.csv")

    pd_T2 = pd.read_csv(T2_value_path)
    T2_value_list = list(pd_T2[T2_name])
    get_value = GetRValue(feature_path, T2_value_list, T2_name)
    get_value.get_R_P_feature_name()


def show_correlation(csv_path=None):
    """展示相关系数图"""
    csv_path = r"E:\Data\doctor_xie\Feature\max_correlation.csv"
    pd_csv = pd.read_csv(csv_path)
    name = list(pd_csv.columns)
    feature_name = name[0]
    T2_name = name[1]

    feature_value = pd_csv[feature_name]
    T2_value = pd_csv[T2_name]
    sns.regplot(x=T2_value, y=feature_value)
    plt.xlabel("T2_value")
    plt.show()


if __name__ == '__main__':
    root_path = r'Y:\DYB\data_and_result\doctor_xie\knee_data'
    store_path = r"Y:\DYB\data_and_result\doctor_xie\Feature"  # 存放T2值的路径
    # run(root_path)  # 得到 软骨下骨 单独的四个roi
    # run_check(root_path)

    # get_T2(root_path, store_path)  # 得到T2值
    """mean_up_1	mean_up_2	mean_down_1	mean_down_2	ninety_up_1	ninety_up_2	ninety_down_1	ninety_down_2"""

    get_R_P(feature_name="xia_gu_up_1", T2_name="mean_up_1")
    get_R_P(feature_name="xia_gu_up_1", T2_name="ninety_up_1")
    get_R_P(feature_name="xia_gu_up_2", T2_name="mean_up_2")
    get_R_P(feature_name="xia_gu_up_2", T2_name="ninety_up_2")
    get_R_P(feature_name="xia_gu_down_1", T2_name="mean_down_1")
    get_R_P(feature_name="xia_gu_down_1", T2_name="ninety_down_1")
    get_R_P(feature_name="xia_gu_down_2", T2_name="mean_down_2")
    get_R_P(feature_name="xia_gu_down_2", T2_name="ninety_down_2")

    """展示相关性图"""
    # show_correlation()