import ants
import os
import pydicom
import numpy as np
import scipy.io as sio
import SimpleITK as sitk
import matplotlib.pyplot as plt
from sklearn import metrics as mr


class preprocessing():
    def __init__(self):
        super(preprocessing, self).__init__()

    def data_preprocessing(self, ep):
        save_warp_path = f"C:/Users/cgqzsyc/Desktop/ValidationSet(Visualization)(0924)/{ep}/{ep}f.dcm"
        save_fix_path = f"C:/Users/cgqzsyc/Desktop/ValidationSet(Visualization)(0924)/{ep}/{ep}g.dcm"
        # moving image (aca)
        path_mov = f"C:/Users/cgqzsyc/Desktop/ValidationSet(Visualization)(0924)/{ep}/{ep}FAST.dcm"  # moving(fast) img's folder
        itk_mov = sitk.ReadImage(path_mov)
        img_mov = sitk.GetArrayFromImage(itk_mov).astype(np.float32)
        img_mov = img_mov[0, :, :]  # size:[1, ., .]->[., .]
        scale_mov = np.mean(img_mov)
        img_mov = img_mov / scale_mov
        # dicom head
        py_dcm_mov = pydicom.read_file(path_mov)

        path_fix = f"C:/Users/cgqzsyc/Desktop/ValidationSet(Visualization)(0924)/{ep}/{ep}GT.dcm"
        itk_fix = sitk.ReadImage(path_fix)
        img_fix = sitk.GetArrayFromImage(itk_fix).astype(np.float32)
        img_fix = img_fix[0, :, :]
        scale_fix = np.mean(img_fix)
        img_fix = img_fix / scale_fix
        # dicom head
        py_dcm_fix = pydicom.read_file(path_fix)
        # print("1")

        nx_fix = img_fix.shape[0]
        nx_mov = img_mov.shape[0]

        # debug output: if nx_fix != nx_mov:
        if nx_mov != nx_fix:
            print('Original moving image size: %d x %d' % (
                img_mov.shape[0], img_mov.shape[1]))
            print('Original fix image size: %d x %d' % (img_fix.shape[0], img_fix.shape[1]))

        if nx_mov < nx_fix:
            print('Padding...')
            k_mov = np.fft.fftshift(
                np.fft.fftn(
                    np.fft.ifftshift(
                        img_mov, axes=(0, 1)
                    ), axes=(0, 1), norm='ortho'
                ), axes=(0, 1)
            )
            # zero-padding
            pad1 = (nx_fix >> 1) - (nx_mov >> 1)
            pad2 = nx_fix - nx_mov - pad1
            pz1 = np.zeros((pad1, nx_mov), dtype=k_mov.dtype)
            pz2 = np.zeros((pad2, nx_mov), dtype=k_mov.dtype)
            k_mov = np.concatenate((pz1, k_mov, pz2), axis=0)
            pz1 = np.zeros((nx_fix, pad1), dtype=k_mov.dtype)
            pz2 = np.zeros((nx_fix, pad2), dtype=k_mov.dtype)
            k_mov = np.concatenate((pz1, k_mov, pz2), axis=1)
            img_mov = np.fft.fftshift(
                np.fft.ifftn(
                    np.fft.ifftshift(
                        k_mov, axes=(0, 1)
                    ), axes=(0, 1), norm='ortho'
                ), axes=(0, 1)
            )
            img_mov = np.abs(img_mov).astype(img_fix.dtype)
            print('Moving image size after padding: %d x %d' % (
                img_mov.shape[0], img_mov.shape[1]))
        elif nx_mov > nx_fix:
            print('Cropping...')
            k_mov = np.fft.fftshift(
                np.fft.fftn(
                    np.fft.ifftshift(
                        img_mov, axes=(0, 1)
                    ), axes=(0, 1), norm='ortho'
                ), axes=(0, 1)
            )
            # crop
            crop1 = (nx_mov >> 1) - (nx_fix >> 1)
            crop2 = crop1 + nx_fix
            k_mov = k_mov[crop1:crop2, crop1:crop2]
            img_mov = np.fft.fftshift(
                np.fft.ifftn(
                    np.fft.ifftshift(
                        k_mov, axes=(0, 1)
                    ), axes=(0, 1), norm='ortho'
                ), axes=(0, 1)
            )
            img_mov = np.abs(img_mov).astype(img_fix.dtype)
            print('Moving image size after cropping: %d x %d' % (
                img_mov.shape[0], img_mov.shape[1]))

        # image registration
        img_fix = img_fix.astype(np.float32)
        img_mov = img_mov.astype(np.float32)

        fixed_image = ants.from_numpy(img_fix, has_components=False)
        moving_image = ants.from_numpy(img_mov, has_components=False)

        out = ants.registration(fixed_image, moving_image, type_of_transform='DenseRigid')  # DenseRigid
        img_warp = out['warpedmovout'].numpy()

        img_warp *= scale_mov
        img_fix *= scale_fix

        print(img_warp.shape, img_warp.max(), img_warp.min())

        img_warp = img_warp[np.newaxis, :, :]
        img_fix = img_fix[np.newaxis, :, :]
        im_warp_save = np.abs(img_warp)
        im_warp_save[im_warp_save > 65535] = 65535
        im_warp_save = im_warp_save.astype(np.uint16)
        if nx_mov != nx_fix:
            tmp = py_dcm_mov.SeriesDescription
            py_dcm_mov = pydicom.read_file(
                path_fix)  # use the matrix size and fov of fixing image
            py_dcm_mov.SOPInstanceUID += '1'
            py_dcm_mov.StudyInstanceUID += '1'
            py_dcm_mov.SeriesInstanceUID += '1'
            py_dcm_mov.SeriesDescription = tmp
            py_dcm_mov.PixelData = im_warp_save.tobytes()
        else:
            py_dcm_mov.PixelData = im_warp_save.tobytes()

        im_fix_save = np.abs(img_fix)
        im_fix_save[im_fix_save > 65535] = 65535
        im_fix_save = im_fix_save.astype(np.uint16)
        py_dcm_fix.PixelData = im_fix_save.tobytes()
        mi_score = mr.normalized_mutual_info_score(
            np.reshape(im_fix_save, (-1,)), np.reshape(im_warp_save, (-1,))
        )
        print('Normalized mutual info: %.5f' % mi_score)
        # print("3")
        if mi_score > 0.29:
            # save results
            py_dcm_mov.save_as(save_warp_path)
            py_dcm_fix.save_as(save_fix_path)
