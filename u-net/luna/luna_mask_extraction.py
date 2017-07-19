#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import SimpleITK as sitk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os

from glob import glob

try:
    from tqdm import tqdm  # long waits are not fun
except ImportError as e:
    print('TQDM does make much nicer wait bars...')
    tqdm = lambda x: x

luna_subset_path = '/Onepiece/project/PythonProject/Lazy/tensorflow/data/luna/subset0/'
luna_path = '/Onepiece/project/PythonProject/Lazy/tensorflow/data/luna/CSVFILES/'
output_path = '/Onepiece/project/PythonProject/Lazy/tensorflow/data/luna/output_data/'
file_list = glob(luna_subset_path + "*.mhd")


def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return f


def make_mask(center, diam, z, width, height, spacing, origin):
    mask = np.zeros([height, width])  # 0's everywhere except nodule swapping x,y to match img
    # convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    v_center = (center - origin) / spacing
    v_diam = int(diam / spacing[0] + 5)

    v_xmin = np.max([0, int(v_center[0] - v_diam) - 5])
    v_xmax = np.min([width - 1, int(v_center[0] + v_diam) + 5])
    v_ymin = np.max([0, int(v_center[1] - v_diam) - 5])
    v_ymax = np.min([height - 1, int(v_center[1] + v_diam) + 5])

    v_xrange = range(v_xmin, v_xmax + 1)
    v_yrange = range(v_ymin, v_ymax + 1)

    # Convert back to world coordinates for distance calculation
    x_data = [x * spacing[0] + origin[0] for x in range(width)]
    y_data = [x * spacing[1] + origin[1] for x in range(height)]

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0] * v_x + origin[0]
            p_y = spacing[1] * v_y + origin[1]
            if np.linalg.norm(center - np.array([p_x, p_y, z])) <= diam:
                mask[int((p_y - origin[1]) / spacing[1]), int((p_x - origin[0]) / spacing[0])] = 1.0
    return mask


def matrix2int16(matrix):
    m_min = np.min(matrix)
    m_max = np.max(matrix)
    matrix -= m_min
    return np.array(np.rint((matrix - m_min) / float(m_max - m_min) * 65535.0), dtype=np.uint16)


def main():
    df_node = pd.read_csv(luna_path + "annotations.csv")
    df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
    df_node = df_node.dropna()

    for fcount, img_file in enumerate(tqdm(file_list)):
        mini_df = df_node[df_node["file"] == img_file]  # get all nodules associate with file
        if mini_df.shape[0] > 0:  # some files may not have a nodule--skipping those
            # load the data once
            itk_img = sitk.ReadImage(img_file)
            img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
            num_z, height, width = img_array.shape  # heightXwidth constitute the transverse plane
            origin = np.array(itk_img.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
            spacing = np.array(itk_img.GetSpacing())  # spacing of voxels in world coor. (mm)

            # go through all nodes (why just the biggest?)
            for node_idx, cur_row in mini_df.iterrows():
                node_x = cur_row["coordX"]
                node_y = cur_row["coordY"]
                node_z = cur_row["coordZ"]
                diam = cur_row["diameter_mm"]
                # just keep 3 slices
                imgs = np.ndarray([3, height, width], dtype=np.float32)
                masks = np.ndarray([3, height, width], dtype=np.uint8)
                center = np.array([node_x, node_y, node_z])  # nodule center
                v_center = np.rint((center - origin) / spacing)  # nodule center in voxel space (still x,y,z ordering)
                for i, i_z in enumerate(np.arange(int(v_center[2]) - 1,
                                                  int(v_center[2]) + 2).clip(0,
                                                                             num_z - 1)):  # clip prevents going out of bounds in Z
                    mask = make_mask(center, diam, i_z * spacing[2] + origin[2],
                                     width, height, spacing, origin)
                    masks[i] = mask
                    imgs[i] = img_array[i_z]
                np.save(os.path.join(output_path, "images_%d_%d.npy" % (fcount, node_idx)), imgs)
                np.save(os.path.join(output_path, "masks_%d_%d.npy" % (fcount, node_idx)), masks)


def plot_image():
    imgs = np.load(output_path + 'images_1_23.npy')
    masks = np.load(output_path + 'masks_1_23.npy')

    for i in range(len(imgs)):
        print
        "image %d" % i
        fig, ax = plt.subplots(2, 2, figsize=[8, 8])
        ax[0, 0].imshow(imgs[i], cmap='gray')
        ax[0, 1].imshow(masks[i], cmap='gray')
        ax[1, 0].imshow(imgs[i] * masks[i], cmap='gray')
        plt.show()
        raw_input("hit enter to cont : ")


if __name__ == '__main__':
    # main()
    plot_image()
