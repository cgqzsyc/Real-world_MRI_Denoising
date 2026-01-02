import os
import torch
import torch.nn as nn
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import pydicom._storage_sopclass_uids

class savedicom():
    def __init__(self):
        super(savedicom, self).__init__()

    def save_as_dicom(self, x_i_result, path, py_dcm_origin):
        # x_i_result = x_i_result.detach().cpu().numpy()  # [nx, ny]
        # x_i_result = np.abs(x_i_result)
        # x_i_result[x_i_result > 65535] = 65535
        # x_i_result = x_i_result.astype(np.uint16)
        # py_dcm_save = py_dcm_origin
        # py_dcm_save.PixelData = x_i_result.tobytes()
        # # py_dcm_save.SOPInstanceUID += '10302'  # these UIDs should be different from output dcm files
        # # py_dcm_save.StudyInstanceUID += '10302'
        # # py_dcm_save.SeriesInstanceUID += '10302'
        # # py_dcm_save.Rows = x_i_result.shape[0]
        # # py_dcm_save.Columns = x_i_result.shape[1]
        # py_dcm_save.save_as(path)
        import numpy as np, copy
        from pydicom.uid import ExplicitVRLittleEndian

        arr = x_i_result.detach().cpu().numpy()
        arr = np.abs(arr)
        arr = np.clip(arr, 0, 65535).astype(np.uint16)
        rows, cols = int(arr.shape[0]), int(arr.shape[1])

        ds = copy.deepcopy(py_dcm_origin)

        need_force_uncompressed = False
        try:
            need_force_uncompressed = bool(ds.file_meta.TransferSyntaxUID.is_compressed)
        except Exception:
            need_force_uncompressed = True

        if need_force_uncompressed:
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds.is_little_endian = True
            ds.is_implicit_VR = False
            for tag in ('LossyImageCompression', 'DerivationDescription', 'LossyImageCompressionRatio'):
                if tag in ds:
                    del ds[tag]

        if getattr(ds, 'Rows', None) != rows:
            ds.Rows = rows
        if getattr(ds, 'Columns', None) != cols:
            ds.Columns = cols

        if 'NumberOfFrames' in ds:
            try:
                if int(ds.NumberOfFrames) != 1:
                    ds.NumberOfFrames = "1"
            except Exception:
                ds.NumberOfFrames = "1"

        if not hasattr(ds, 'SamplesPerPixel'):
            ds.SamplesPerPixel = 1
        if not hasattr(ds, 'PhotometricInterpretation'):
            ds.PhotometricInterpretation = 'MONOCHROME2'

        ba = getattr(ds, 'BitsAllocated', None)
        bs = getattr(ds, 'BitsStored', None)
        pr = getattr(ds, 'PixelRepresentation', None)

        if ba != 16:
            ds.BitsAllocated = 16
        if bs is None or bs > getattr(ds, 'BitsAllocated', 16) or bs < 1:
            ds.BitsStored = 16
        ds.HighBit = ds.BitsStored - 1

        if pr is None:
            ds.PixelRepresentation = 0
        else:
            if pr == 1 and int(arr.max()) > 32767:
                ds.PixelRepresentation = 0

        ds.PixelData = arr.tobytes()
        if 'PixelData' in ds:
            try:
                ds['PixelData'].is_undefined_length = False
            except Exception:
                pass

        ds.save_as(path, write_like_original=False)
        '''
        x_i_result_np = x_i_result.detach().cpu().numpy().astype(np.uint16)
        # print(x_i_result_np.shape)
        meta = pydicom.Dataset()
        meta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.MRImageStorage
        meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        ds = Dataset()
        ds.file_meta = meta

        ds.is_little_endian = True
        ds.is_implicit_VR = False

        ds.SOPClassUID = pydicom._storage_sopclass_uids.MRImageStorage
        ds.PatientName = "Test^Firstname"
        ds.PatientID = "123456"

        ds.Modality = "MR"
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
        ds.FrameOfReferenceUID = pydicom.uid.generate_uid()

        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.SamplesPerPixel = 1
        ds.HighBit = 15

        ds.ImagesInAcquisition = "1"

        ds.Rows = x_i_result_np.shape[0]
        ds.Columns = x_i_result_np.shape[1]
        ds.InstanceNumber = 1

        ds.ImagePositionPatient = r"0\0\1"
        ds.ImageOrientationPatient = r"1\0\0\0\-1\0"
        ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"

        ds.RescaleIntercept = "0"
        ds.RescaleSlope = "1"
        ds.PixelSpacing = r"1\1"
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 1

        pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)
        ds.PixelData = x_i_result_np.tobytes()

        ds.save_as(file_name)
        # ds.save_as(f"./result/x_denoise_result_{ep}_{i}.dcm")
        # np.save(f"./result/x_denoise_result_{ep}.npy", x_gen)  # 如果之后改成保存numpy，则保存文件名后缀名相应改为.npy
        # np.save(f"./result/x_denoise_tmp_result_{ep}.npy", x_gen_store)
        '''

    def norm_inv(self, x, minx, maxx):
        x = x.squeeze()
        x_i = (x + 1) * (maxx - minx) / 2.0 + minx
        # x_i = x * (maxx - minx) + minx
        x_i = torch.floor_(x_i)
        x_i = torch.clamp(x_i, 0)
        return x_i