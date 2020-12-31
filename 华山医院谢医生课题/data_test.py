import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from MeDIT.Visualization import Imshow3DArray
from jia_segment.loss_function import tversky_loss
from scipy import ndimage
import warnings
warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class ModelTester:
    def __init__(self, test_data_folder, image_name, roi_name, STD_IMAGE_SHAPE, trained_model):
        self.test_data_folder = test_data_folder
        self.image_name = image_name
        self.roi_name = roi_name
        self.STD_IMAGE_SHAPE = STD_IMAGE_SHAPE
        self.trained_model = trained_model

    def create_roi(self, pred_roi):
        pred_roi = pred_roi[0, :, :, 0]
        pred_roi = np.where(pred_roi > 0.999, 1., 0.)  # 对概率设置阈值
        pred_roi = pred_roi.astype(np.float64)
        return pred_roi

    def dice_coef(self, y_true, y_pred):
        smooth = 1.  # 用于防止分母为0.
        y_true_f = y_true.ravel()  # 将 y_true 拉伸为一维.
        y_pred_f = y_pred.ravel()
        intersection = np.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (np.sum(y_true_f * y_true_f) + np.sum(y_pred_f * y_pred_f) + smooth)

    @staticmethod
    def RemoveSmallRegion(mask):
        label_im, nb_labels = ndimage.label(mask)
        p = []
        for i in range(1, nb_labels + 1):
            c = (label_im == i).sum()
            p.append(c)
        size_thres_all = sorted(p)
        size_thres = 100000000
        if len(size_thres_all) == 0:
            pass
        elif len(size_thres_all) == 1:
            size_thres = max(size_thres_all)
        elif len(size_thres_all) > 1:
            del (size_thres_all[-1])
            size_thres = max(size_thres_all)
        for i in range(1, nb_labels + 1):
            if (label_im == i).sum() < size_thres:
                # remove the small ROI in mask
                mask[label_im == i] = 0
        print(max(p))
        print(size_thres)
        return mask

    def remove_small_roi(self, mask):
        label_im, nb_labels = ndimage.label(mask)
        print(nb_labels)

    def compute_3d_dice(self):
        dice_list = []
        for folder in os.listdir(self.test_data_folder):
            image = np.load(os.path.join(self.test_data_folder, folder, self.image_name))
            roi = np.load(os.path.join(self.test_data_folder, folder, self.roi_name))
            pred_roi_list = []
            for index in range(image.shape[-1]):  # 根据测试数据的层数进行索引
                # if index == 0 or index == image.shape[-1]:
                #     pred_roi_list.append(np.zeros(self.STD_IMAGE_SHAPE))
                # else:
                image_slice = image[:, :, index]
                image_slice_crop = np.expand_dims(image_slice[:, :], axis=2)
                image_slice_crop = np.expand_dims(image_slice_crop[:, :], axis=0)
                image_slice_crop = (image_slice_crop - np.mean(image_slice_crop)) / np.std(image_slice_crop)
                pred_roi = self.trained_model.predict(image_slice_crop)
                one_slice = self.create_roi(pred_roi)
                pred_roi_list.append(one_slice)

            pred_roi_list = np.array(pred_roi_list).transpose((1, 2, 0))
            pred_roi_list = self.RemoveSmallRegion(pred_roi_list)  # 这是预测的ROI
            # pred_roi_list = self.remove_small_roi(pred_roi_list)
            # 后续直接保存数据即可
            # np.save(os.path.join(self.test_data_folder, folder, "pred_roi.npy"), pred_roi_list)
            print("case {} is finished!".format(folder))
            # for i in range(image.shape[-1]):
            #     if np.sum(roi) != 0:
            #         plt.imshow(image[..., i], cmap="gray")
            #         plt.contour(roi[..., i])
            #         plt.contour(pred_roi_list[..., i], colors="red")
            #         plt.show()
            image = (image - np.min(image)) / ((np.max(image) - np.min(image)))
            image_show = image
            dice = self.dice_coef(roi, pred_roi_list)
            # Imshow3DArray(image_show, [roi, pred_roi_list])

            print(image_show.shape, roi.shape, pred_roi_list.shape)
            print("current case:", folder)
            print("current case dice is:", round(dice, 3))
            dice_list.append(round(dice, 4))
        return dice_list


if __name__ == "__main__":
    # 调用需要的参数和文件路径
    test_data_folder = r'Y:\DYB\2020832DATA\doctor_xie\normal_control'
    model_path = r"Y:\DYB\2020832DATA\doctor_xie\model\roi_1.hdf5"
    shape = (384, 384)

    train_model = tf.keras.models.load_model(model_path, custom_objects={
        'tversky_loss': tversky_loss})
    # 调用函数
    tester = ModelTester(test_data_folder=test_data_folder, image_name='data.npy',
                         roi_name='roi_1.npy', STD_IMAGE_SHAPE=shape, trained_model=train_model)
    all_case_dice = tester.compute_3d_dice()
    plt.hist(all_case_dice, edgecolor="black")
    plt.show()
    # 输出结果进行查看
    print('roi:', all_case_dice)
    print('mean:', round(float(np.mean(all_case_dice)), 4))
    print('std:', round(float(np.std(all_case_dice)), 4))
    print('max:', np.max(all_case_dice))
    print('min:', np.min(all_case_dice))
