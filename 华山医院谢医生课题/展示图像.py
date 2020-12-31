import os
from MeDIT.Visualization import Imshow3DArray
from Tool.DataProcress import get_array_from_path, standard
import numpy as np

data = get_array_from_path(r"Y:\DYB\data_and_result\doctor_xie\knee_data\CAI NAI JI-20140605\data_1.nii")
data = standard(data)
data = np.flipud(data)
roi = get_array_from_path(r"Y:\DYB\data_and_result\doctor_xie\knee_data\CAI NAI JI-20140605\xia_gu_down_2.nii")
roi_2 = get_array_from_path(r"Y:\DYB\data_and_result\doctor_xie\knee_data\CAI NAI JI-20140605\ruan_gu_down_2.nii")
Imshow3DArray(data, [roi, roi_2])
Imshow3DArray(data, roi_2)
Imshow3DArray(data, roi)