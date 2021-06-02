import numpy as np

USTC_vis_data = np.load(r'E:\Work\Multi Modal Face Recognition\Numpy Data\USTC Data\USTC CMFR Train Vis Images.npy')
USTC_the_data = np.load(r'E:\Work\Multi Modal Face Recognition\Numpy Data\USTC Data\USTC CMFR Train The Images.npy')
USTC_lbl = np.load(r'E:\Work\Multi Modal Face Recognition\Numpy Data\USTC Data\USTC CMFR Train Labels.npy')

TUFS_vis_data = np.load(r'E:\Work\Multi Modal Face Recognition\Numpy Data\TUFS Data\TUFS CMFR Train Vis Images.npy')
TUFS_the_data = np.load(r'E:\Work\Multi Modal Face Recognition\Numpy Data\TUFS Data\TUFS CMFR Train The Images.npy')
TUFS_lbl = np.load(r'E:\Work\Multi Modal Face Recognition\Numpy Data\TUFS Data\TUFS CMFR Train Labels.npy')

print('Done')