import os
import numpy as np
import math
import random
from MeDIT import DataAugmentor

from MeDIT.Visualization import Imshow3DArray

import warnings
from ankle_code.att_model import attn_unet

warnings.filterwarnings("ignore")


def transform_data_to_dict(root_path, roi_name, dict_save_name):
    roi_dict = {}
    for folder in os.listdir(root_path):
        roi = np.load(os.path.join(root_path, folder, roi_name))
        roi_sum = np.sum(roi, axis=(0, 1))  # 数据加起来形成一个层方向上的数据
        roi_dict[folder] = roi_sum
    np.save(dict_save_name, roi_dict)
    return roi_dict


def chose_roi_index(roi_sum, rate=0.3):
    min_index, max_index = 0, roi_sum.shape[-1] - 1
    temp_list = list(range(roi_sum.shape[-1]))
    temp_list.remove(min_index)
    temp_list.remove(max_index)
    index_list = list(np.where(roi_sum != 0)[0])
    if min_index in index_list:
        index_list.remove(min_index)
    if max_index in index_list:
        index_list.remove(max_index)
    not_tumor_list = list(set(temp_list) - set(index_list))
    no_tumor_number = int(math.floor(len(not_tumor_list) * rate))
    no_sample = random.sample(not_tumor_list, no_tumor_number)
    return index_list + no_sample


def get_data_list(data_dict):
    index_with_name = []
    for name in data_dict:
        roi_index = data_dict[name]
        index_set = chose_roi_index(roi_sum=roi_index)
        for index_slice in index_set:
            index_with_name.append((name, index_slice))
    return index_with_name  # len(index_with_name) = 355


def aug_data(image, roi, stretch_x, stretch_y, shear, rotate_z_angle, shift_x, shift_y):
    random_params = {'stretch_x': stretch_x, 'stretch_y': stretch_y, 'shear': shear,
                     'rotate_z_angle': rotate_z_angle, 'shift_x': shift_x, 'shift_y': shift_y}
    param_generator = DataAugmentor.AugmentParametersGenerator()
    aug_generator = DataAugmentor.DataAugmentor2D()
    param_generator.RandomParameters(random_params)
    aug_generator.SetParameter(param_generator.GetRandomParametersDict())
    new_data = aug_generator.Execute(image)
    new_roi = aug_generator.Execute(roi)
    return new_data, new_roi


def get_batch_data(data_list, batch_size):
    range_time = len(data_list) // batch_size
    for i in range(range_time):
        yield data_list[i*batch_size:(i+1) * batch_size]


def get_batch_block(folder_path, data_dict, batch_size, big_size, STD_IMAGE_SHAPE, image_name, roi_name):
    while True:
        index_with_name = get_data_list(data_dict)
        random.shuffle(index_with_name)
        batch_inter = get_batch_data(data_list=index_with_name, batch_size=batch_size)
        for batch_data in batch_inter:
            data_block = np.zeros((0, STD_IMAGE_SHAPE[0], STD_IMAGE_SHAPE[1], 3))
            roi_block = np.zeros((0, STD_IMAGE_SHAPE[0], STD_IMAGE_SHAPE[1], 9))

            for case_name, roi_index in batch_data:
                case_path = os.path.join(folder_path, case_name)
                image = os.path.join(case_path, "resized_data.npy")
                roi_1 = os.path.join(case_path, "resized_roi_1.npy")
                roi_2 = os.path.join(case_path, "resized_roi_2.npy")
                roi_3 = os.path.join(case_path, "resized_roi_3.npy")
                roi_4 = os.path.join(case_path, "resized_roi_4.npy")
                roi_5 = os.path.join(case_path, "resized_roi_5.npy")
                roi_6 = os.path.join(case_path, "resized_roi_6.npy")
                roi_7 = os.path.join(case_path, "resized_roi_7.npy")
                roi_8 = os.path.join(case_path, "resized_roi_8.npy")
                image = np.load(image)
                roi_1 = np.load(roi_1)
                roi_2 = np.load(roi_2)
                roi_3 = np.load(roi_3)
                roi_4 = np.load(roi_4)
                roi_5 = np.load(roi_5)
                roi_6 = np.load(roi_6)
                roi_7 = np.load(roi_7)
                roi_8 = np.load(roi_8)


                image = image[:, :, roi_index-1:roi_index+2]

                roi_1 = np.expand_dims(roi_1[:, :, roi_index], axis=2)
                roi_2 = np.expand_dims(roi_2[:, :, roi_index], axis=2)
                roi_3 = np.expand_dims(roi_3[:, :, roi_index], axis=2)
                roi_4 = np.expand_dims(roi_4[:, :, roi_index], axis=2)
                roi_5 = np.expand_dims(roi_5[:, :, roi_index], axis=2)
                roi_6 = np.expand_dims(roi_6[:, :, roi_index], axis=2)
                roi_7 = np.expand_dims(roi_7[:, :, roi_index], axis=2)
                roi_8 = np.expand_dims(roi_8[:, :, roi_index], axis=2)

                background_roi = np.ones((STD_IMAGE_SHAPE[0], STD_IMAGE_SHAPE[1], 1)) - roi_1 - \
                                 roi_2 - roi_3 - roi_4 - roi_5 - roi_6 - roi_7 - roi_8
                background_roi = np.where(background_roi <= 0., 0., 1.)
                roi_data = np.concatenate((background_roi, roi_1, roi_2, roi_3, roi_4, roi_5, roi_6, roi_7, roi_8),
                                          axis=2)

                roi_data_new = np.expand_dims(roi_data, axis=0)
                image = (image - np.mean(image)) / (np.std(image))
                image = np.expand_dims(image, axis=0)

                roi_block = np.append(roi_block, roi_data_new, axis=0)
                data_block = np.append(data_block, image, axis=0)
                yield data_block, roi_block



def train_model(train_dict_path, val_dict_path, batch_size, epochs):
    callbacks = [
        TensorBoard(),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, mode='min'),
        EarlyStopping(monitor='val_loss', patience=100, mode='min', min_delta=0.0001),
        ModelCheckpoint(filepath='RESULT/valloss{val_loss:.3f}_loss{loss:.3f}_brain_epoch{epoch:03d}.hdf5',
                        monitor='val_loss',
                        save_best_only=False, mode='min', period=1)]

    # model = unet()
    input_size = (160, 160, 3)
    model = attn_unet(input_size)
    model.compile(optimizer=Adam(lr=1e-4), loss=tversky_loss)

    train_dict = np.load(train_dict_path, allow_pickle=True).item()
    val_dict = np.load(val_dict_path, allow_pickle=True).item()
    steps_per_epoch = len(get_data_list(train_dict)) // batch_size
    val_steps = len(get_data_list(val_dict)) // batch_size
    # print(val_steps)

    train_generator = get_batch_block(folder_path=train_folder_path, data_dict=train_roi_dict, batch_size=batch_size,
                                      big_size=big_size, STD_IMAGE_SHAPE=STD_IMAGE_SHAPE, image_name=image_name,
                                      roi_name=roi_name)
    validation_generator = get_batch_block(folder_path=val_folder_path, data_dict=val_roi_dict,
                                           batch_size=batch_size,
                                           big_size=big_size, STD_IMAGE_SHAPE=STD_IMAGE_SHAPE, image_name=image_name,
                                           roi_name=roi_name)
    history = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                  validation_data=validation_generator,
                                  validation_steps=val_steps,
                                  callbacks=callbacks)


if __name__ == "__main__":
    # 调用需要的参数和文件的路径
    train_folder_path = r"/home/danyibo/data/ankle_data/train/"
    val_folder_path = r"/home/danyibo/data/ankle_data/val/"
    roi_name = r"resized_all_roi.npy"
    image_name = r"resized_data.npy"
    STD_IMAGE_SHAPE = [160, 160]
    big_size = [STD_IMAGE_SHAPE[0] + 10, STD_IMAGE_SHAPE[1] + 10]
    train_dict_path = r"train_index.npy"
    val_dict_path = r"val_index.npy"
    batch_size = 1
    epochs = 500000

    # 开始调用函数
    train_roi_dict = transform_data_to_dict(root_path=train_folder_path, roi_name=roi_name,
                                            dict_save_name="train_index.npy")
    val_roi_dict = transform_data_to_dict(root_path=val_folder_path, roi_name=roi_name,
                                          dict_save_name="val_index.npy")
    get_data_list(data_dict=train_roi_dict)
    get_batch_block(folder_path=train_folder_path, data_dict=train_roi_dict, batch_size=batch_size,
                    big_size=big_size, STD_IMAGE_SHAPE=STD_IMAGE_SHAPE, image_name=image_name,
                    roi_name=roi_name)
    train_model(train_dict_path=train_dict_path, val_dict_path=val_dict_path,
                batch_size=batch_size, epochs=epochs)

