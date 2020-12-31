import os
import numpy as np
from MeDIT.Visualization import Imshow3DArray
from Tool.DataProcress import get_array_from_path


root_path = r"Y:\DYB\data_and_result\doctor_xie\knee_data"
for case in os.listdir(root_path):
    T2_path = os.path.join(root_path, case, "T2.nii")
    T2_array = get_array_from_path(T2_path)
    roi_path = os.path.join(root_path, case, "xia_gu_up_1.nii")
    roi_array = get_array_from_path(roi_path)
    Imshow3DArray(T2_array, roi_array)