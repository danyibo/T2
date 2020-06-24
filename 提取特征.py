import os
import numpy as np
import SimpleITK as sitk
from MeDIT.Visualization import Imshow3DArray
from MeDIT.SaveAndLoad import SaveNumpyToImageByRef
from Ankle.tool import DataProcess as td


class GetRoi:

    def __init__(self, case_path):
        self.case_path = case_path
        self.roi_path = os.path.join(case_path, "roi.nii.gz")
        self.data_path = os.path.join(case_path, "data_1.nii")
        self.roi = sitk.ReadImage(self.roi_path)
        self.data = sitk.ReadImage(self.data_path)

    def move_roi(self, roi_array):
        roi_shape = roi_array.shape
        moved_value = 12
        roi_big = roi_array[:roi_shape[0]-moved_value, :, :]
        roi_small = roi_array[roi_shape[0]-moved_value:, :, :]
        new_roi_array = np.vstack((roi_small, roi_big))
        return new_roi_array

    def save_moved_roi(self):
        roi_array = td.get_array(self.roi_path)
        roi_4_array = np.where(roi_array == 4, 1, 0)
        data_array = td.get_array(self.data_path)
        data_array = td.standard(data_array)
        data_array = np.flipud(data_array)
        new_roi_array = self.move_roi(roi_4_array)
        # SaveNumpyToImageByRef(store_path=os.path.join(self.case_path, "moved_roi_4.nii"), data=new_roi_array,
        #                       ref_image=self.data)
        Imshow3DArray(data_array, [roi_4_array, new_roi_array])


def move_all_roi(all_data_path):
    for case in os.listdir(all_data_path):
        case_path = os.path.join(all_data_path, case)
        get_roi = GetRoi(case_path=case_path)
        get_roi.save_moved_roi()


if __name__ == "__main__":
    move_all_roi(all_data_path=r'E:\split_data_ACLR_T2')