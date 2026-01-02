import os
import sys
import torch
import numpy as np
import random
import pickle
import pydicom
import SimpleITK as sitk
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import crop
from torchvision import transforms
from PIL import Image


class MriTrainConDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        self.target = torch.empty((0,))
        self.map_organ = {'BRAIN': 1, 'HEAD': 1, 'KNEE': 2, 'CSPINE': 3, 'LSPINE': 4, 'TSPINE': 5,
                          'SPINE': 6, 'CAROTID': 7, 'SHOULDER': 8, 'UTERUS': 9, 'lp': 10,
                          'EMPTY': 0}
        self.uii_acc_map = {'FULL': 8, 'ACS2': 2, 'ACS3': 3, 'ACS5': 5, 'ACS7': 7,
                            'PPA2': 2, 'PPA3': 3}

        tot = 0
        data_root = data_path + '/15T_GE_MR'

        date_folders = os.listdir(data_root)
        for date_folder in date_folders:  # root folder
            patient_folders = os.listdir(os.path.join(data_root, date_folder))
            for patient_folder in patient_folders:  # patient folder
                organ_folders = os.listdir(os.path.join(data_root, date_folder, patient_folder))
                for organ_folder in organ_folders:  # organ folder
                    seq_folders = os.listdir(os.path.join(data_root, date_folder, patient_folder, organ_folder))
                    for seq_folder in seq_folders:  # seq folder
                        if 'FAST' in seq_folder:
                            path_partial_aca = os.path.join(
                                data_root, date_folder, patient_folder, organ_folder, seq_folder
                            )

                            seq_folder_comp = seq_folder.split('_')
                            seq_folder_comp_new = []
                            for comp_id in range(len(seq_folder_comp)):
                                if not 'FAST' in seq_folder_comp[comp_id]:
                                    seq_folder_comp_new.append(seq_folder_comp[comp_id])
                            seq_gt_folder = '_'.join(seq_folder_comp_new)
                            path_partial_gt = os.path.join(
                                data_root, date_folder, patient_folder, organ_folder, seq_gt_folder
                            )

                            slice_names_aca = os.listdir(path_partial_aca)
                            for slice_name in slice_names_aca:
                                target_cur = torch.zeros(11)
                                path_mov = os.path.join(path_partial_aca, slice_name)
                                itk_mov = sitk.ReadImage(path_mov)
                                img_mov = sitk.GetArrayFromImage(itk_mov).astype(np.float32)
                                img_mov = img_mov[0, :, :]  # shape: 256x256
                                py_dcm_mov = pydicom.read_file(path_mov)

                                path_fix = os.path.join(path_partial_gt, slice_name)
                                itk_fix = sitk.ReadImage(path_fix)
                                img_fix = sitk.GetArrayFromImage(itk_fix).astype(np.float32)
                                img_fix = img_fix[0, :, :]
                                py_dcm_fix = pydicom.read_file(path_fix)

                                target_cur[0] = 1
                                for organ_name, value in self.map_organ.items():
                                    if organ_name in path_mov:
                                        target_cur[1] = value
                                        break

                                ts_mov, ts_fix = torch.from_numpy(img_mov), torch.from_numpy(img_fix)
                                min_m, max_m = torch.min(ts_mov), torch.max(ts_mov)
                                min_f, max_f = torch.min(ts_fix), torch.max(ts_fix)
                                ts_mov = 2.0 * (ts_mov - min_m) / (max_m - min_m)- 1.0
                                ts_fix = 2.0 * (ts_fix - min_f) / (max_f - min_f)- 1.0

                                target_cur[6], target_cur[7], target_cur[8], target_cur[9] = min_m, max_m, min_f, max_f
                                target_cur[10] = tot

                                # finally
                                tot = tot + 1
                                ts_mov = ts_mov.unsqueeze(0)
                                ts_fix = ts_fix.unsqueeze(0)  # 2x256x256
                                img_all = torch.cat((ts_mov, ts_fix), dim=0)
                                self.data.append(img_all)
                                target_cur = target_cur.unsqueeze(0)
                                self.target = torch.cat((self.target, target_cur), dim=0)
        # print(tot)
        # print("GE finished")

        # Philip
        data_root = data_path + '/15T_Philip_MR'
        date_folders = os.listdir(data_root)
        for date_folder in date_folders:
            patient_folders = os.listdir(os.path.join(data_root, date_folder))
            for patient_folder in patient_folders:
                organ_folders = os.listdir(os.path.join(data_root, date_folder, patient_folder))
                for organ_folder in organ_folders:
                    if os.path.exists(os.path.join(data_root, date_folder, patient_folder, organ_folder)):
                        seq_folders = os.listdir(os.path.join(data_root, date_folder, patient_folder, organ_folder))
                        for seq_folder in seq_folders:
                            if 'FAST' in seq_folder:
                                path_partial_aca = os.path.join(
                                    data_root, date_folder, patient_folder, organ_folder, seq_folder
                                )
                                seq_folder_comp = seq_folder.split('_')
                                seq_folder_comp_new = []
                                for comp_id in range(len(seq_folder_comp)):
                                    if not 'FAST' in seq_folder_comp[comp_id]:
                                        seq_folder_comp_new.append(seq_folder_comp[comp_id])
                                seq_gt_folder = '_'.join(seq_folder_comp_new)
                                path_partial_gt = os.path.join(
                                    data_root, date_folder, patient_folder, organ_folder, seq_gt_folder
                                )

                                slice_names_aca = os.listdir(path_partial_aca)
                                for slice_name in slice_names_aca:
                                    target_cur = torch.zeros(11)
                                    path_mov = os.path.join(path_partial_aca, slice_name)
                                    itk_mov = sitk.ReadImage(path_mov)
                                    img_mov = sitk.GetArrayFromImage(itk_mov).astype(np.float32)
                                    img_mov = img_mov[0, :, :]  # shape: 256x256
                                    py_dcm_mov = pydicom.read_file(path_mov)

                                    path_fix = os.path.join(path_partial_gt, slice_name)
                                    itk_fix = sitk.ReadImage(path_fix)
                                    img_fix = sitk.GetArrayFromImage(itk_fix).astype(np.float32)
                                    img_fix = img_fix[0, :, :]
                                    py_dcm_fix = pydicom.read_file(path_fix)

                                    target_cur[0] = 2
                                    for organ_name, value in self.map_organ.items():
                                        if organ_name in path_mov:
                                            target_cur[1] = value
                                            break

                                    tag = (0x0018, 0x0083)
                                    if tag in py_dcm_fix:
                                        if py_dcm_fix[tag].value is not None:
                                            target_cur[5] = float(py_dcm_fix[tag].value)

                                    ts_mov, ts_fix = torch.from_numpy(img_mov), torch.from_numpy(img_fix)
                                    min_m, max_m = torch.min(ts_mov), torch.max(ts_mov)
                                    min_f, max_f = torch.min(ts_fix), torch.max(ts_fix)
                                    ts_mov = 2.0 * (ts_mov - min_m) / (max_m - min_m)- 1.0
                                    ts_fix = 2.0 * (ts_fix - min_f) / (max_f - min_f)- 1.0

                                    target_cur[6], target_cur[7], target_cur[8], target_cur[9] = min_m, max_m, min_f, max_f
                                    target_cur[10] = tot

                                    tot = tot + 1
                                    ts_mov = ts_mov.unsqueeze(0)
                                    ts_fix = ts_fix.unsqueeze(0)  # 2x256x256
                                    img_all = torch.cat((ts_mov, ts_fix), dim=0)
                                    self.data.append(img_all)
                                    target_cur = target_cur.unsqueeze(0)
                                    self.target = torch.cat((self.target, target_cur), dim=0)
        # print(tot)
        # print("Philip finished")

        # Siemens
        data_root = data_path + '/15T_Siemens_MR'
        date_folders = os.listdir(data_root)
        for date_folder in date_folders:  # date_folders:
            organ_folders = os.listdir(os.path.join(data_root, date_folder))
            for organ_folder in organ_folders:
                if os.path.exists(os.path.join(data_root, date_folder, organ_folder)):
                    seq_folders = os.listdir(os.path.join(data_root, date_folder, organ_folder))
                    for seq_folder in seq_folders:
                        if 'aca' in seq_folder:
                            patient_folders = os.listdir(os.path.join(data_root, date_folder, organ_folder, seq_folder))
                            for patient_folder in patient_folders:
                                path_partial_aca = os.path.join(
                                    data_root, date_folder, organ_folder, seq_folder, patient_folder
                                )
                                seq_gt_folder, path_partial_gt = '', ''
                                if '_' in date_folder:
                                    seq_gt_folder = 'GT2_' + seq_folder[5:]
                                    path_partial_gt = os.path.join(
                                        data_root, date_folder, organ_folder, seq_gt_folder, patient_folder
                                    )
                                    if not os.path.exists(path_partial_gt):
                                        seq_gt_folder = 'GT1_' + seq_folder[5:]
                                        path_partial_gt = os.path.join(
                                            data_root, date_folder, organ_folder, seq_gt_folder, patient_folder
                                        )
                                else:
                                    seq_folder_comp = seq_folder.split('_')
                                    seq_folder_comp_new = []
                                    for comp_id in range(len(seq_folder_comp)):
                                        if (not 'aca' in seq_folder_comp[comp_id]) and (
                                        not 'ppa' in seq_folder_comp[comp_id]) and (
                                        not 'av' in seq_folder_comp[comp_id]):
                                            seq_folder_comp_new.append(seq_folder_comp[comp_id])
                                    seq_gt_folder = '_'.join(seq_folder_comp_new)
                                    path_partial_gt = os.path.join(
                                        data_root, date_folder, organ_folder, seq_gt_folder, patient_folder
                                    )

                                slice_names_aca = os.listdir(path_partial_aca)
                                for slice_name in slice_names_aca:
                                    target_cur = torch.zeros(11)
                                    path_mov = os.path.join(path_partial_aca, slice_name)
                                    itk_mov = sitk.ReadImage(path_mov)
                                    img_mov = sitk.GetArrayFromImage(itk_mov).astype(np.float32)
                                    img_mov = img_mov[0, :, :]  # shape: 256x256
                                    py_dcm_mov = pydicom.read_file(path_mov)

                                    path_fix = os.path.join(path_partial_gt, slice_name)
                                    itk_fix = sitk.ReadImage(path_fix)
                                    img_fix = sitk.GetArrayFromImage(itk_fix).astype(np.float32)
                                    img_fix = img_fix[0, :, :]
                                    py_dcm_fix = pydicom.read_file(path_fix)

                                    target_cur[0] = 3
                                    for organ_name, value in self.map_organ.items():
                                        if organ_name in path_mov:
                                            target_cur[1] = value
                                            break

                                    tag = [(0x5200, 0x9229), (0x0018, 0x9115), (0x0018, 0x9069)]
                                    tag_cur = tag[0]
                                    acc__value = 0
                                    dcm_dataset = py_dcm_fix
                                    for i in range(3):
                                        if tag[i] not in dcm_dataset:
                                            break
                                        if i == 2:
                                            dcm_dataset = dcm_dataset[tag[i]].value
                                        else:
                                            dcm_dataset = dcm_dataset[tag[i]]
                                            dcm_dataset = dcm_dataset[0]
                                    if acc__value != 0:
                                        target_cur[3] = float(dcm_dataset)

                                    tag = (0x0018, 0x9073)
                                    if tag in py_dcm_fix:
                                        if py_dcm_fix[tag].value is not None:
                                            target_cur[4] = float(py_dcm_fix[tag].value)

                                    ts_mov, ts_fix = torch.from_numpy(img_mov), torch.from_numpy(img_fix)
                                    min_m, max_m = torch.min(ts_mov), torch.max(ts_mov)
                                    min_f, max_f = torch.min(ts_fix), torch.max(ts_fix)
                                    ts_mov = 2.0 * (ts_mov - min_m) / (max_m - min_m)- 1.0
                                    ts_fix = 2.0 * (ts_fix - min_f) / (max_f - min_f)- 1.0

                                    target_cur[6], target_cur[7], target_cur[8], target_cur[9] = min_m, max_m, min_f, max_f
                                    target_cur[10] = tot

                                    # finally
                                    tot = tot + 1
                                    ts_mov = ts_mov.unsqueeze(0)
                                    ts_fix = ts_fix.unsqueeze(0)  # 2x256x256
                                    img_all = torch.cat((ts_mov, ts_fix), dim=0)
                                    self.data.append(img_all)
                                    target_cur = target_cur.unsqueeze(0)
                                    self.target = torch.cat((self.target, target_cur), dim=0)
        # print(tot)
        # print("Siemens finished")

        # UIH
        data_root = data_path + '/15T_UIH_MR'
        seq_folders = os.listdir(data_root)
        for seq_folder in seq_folders:
            if ('ACS' in seq_folder) or ('PPA' in seq_folder) or ('FULL' in seq_folder):
                seq_folder_gt = seq_folder[:-5]
                path_partial_aca = os.path.join(data_root, seq_folder)
                path_partial_gt = os.path.join(data_root, seq_folder_gt)
                # print(path_partial_aca, path_partial_gt)
                slice_names = os.listdir(path_partial_aca)
                for slice_name in slice_names:
                    target_cur = torch.zeros(11)
                    path_mov = os.path.join(path_partial_aca, slice_name)
                    itk_mov = sitk.ReadImage(path_mov)
                    img_mov = sitk.GetArrayFromImage(itk_mov).astype(np.float32)
                    img_mov = img_mov[0, :, :]  # shape: 256x256
                    # py_dcm_mov = pydicom.read_file(path_mov)

                    path_fix = os.path.join(path_partial_gt, slice_name)
                    itk_fix = sitk.ReadImage(path_fix)
                    img_fix = sitk.GetArrayFromImage(itk_fix).astype(np.float32)
                    img_fix = img_fix[0, :, :]
                    # py_dcm_fix = pydicom.read_file(path_fix)

                    target_cur[0] = 4
                    for organ_name, value in self.map_organ.items():
                        if (organ_name in path_mov) or (organ_name.lower() in path_mov):
                            target_cur[1] = value
                            break

                    for acc_rate_name, value in self.uii_acc_map.items():
                        if acc_rate_name in path_mov:
                            target_cur[3] = value

                    ts_mov, ts_fix = torch.from_numpy(img_mov), torch.from_numpy(img_fix)
                    min_m, max_m = torch.min(ts_mov), torch.max(ts_mov)
                    min_f, max_f = torch.min(ts_fix), torch.max(ts_fix)
                    ts_mov = 2.0 * (ts_mov - min_m) / (max_m - min_m)- 1.0
                    ts_fix = 2.0 * (ts_fix - min_f) / (max_f - min_f)- 1.0

                    target_cur[6], target_cur[7], target_cur[8], target_cur[9] = min_m, max_m, min_f, max_f
                    target_cur[10] = tot

                    # finally
                    tot = tot + 1
                    ts_mov = ts_mov.unsqueeze(0)
                    ts_fix = ts_fix.unsqueeze(0)  # 2x256x256
                    img_all = torch.cat((ts_mov, ts_fix), dim=0)
                    self.data.append(img_all)
                    target_cur = target_cur.unsqueeze(0)
                    self.target = torch.cat((self.target, target_cur), dim=0)
        # print(tot)
        # print("UIH finished")

        print(len(self.data))
        print(self.target.shape)
        print(tot)
        torch.cuda.empty_cache()

    def __len__(self):
        return len(self.data)

    def __shape__(self):
        return self.data.shape

    def __getitem__(self, index):
        img_fast = self.data[index][0]
        img_gt = self.data[index][1]
        padding_y_left = (256 - img_fast.shape[1]) // 2
        padding_y_right = 256 - img_fast.shape[1] - padding_y_left
        padding_x_left = (256 - img_fast.shape[0]) // 2
        padding_x_right = 256 - img_fast.shape[0] - padding_x_left
        img_fast = F.pad(img_fast, (max(0, padding_y_left), max(0, padding_y_right),
                                    max(0, padding_x_left), max(0, padding_x_right)), mode="constant", value=-1)
        img_gt = F.pad(img_gt, (max(0, padding_y_left), max(0, padding_y_right),
                                max(0, padding_x_left), max(0, padding_x_right)), mode="constant", value=-1)
        h, w = img_fast.shape[0], img_fast.shape[1]
        x_i = random.randint(0, h - 256)
        y_i = random.randint(0, w - 256)
        img_f = crop(img_fast, top=x_i, left=y_i, height=256, width=256)
        img_g = crop(img_gt, top=x_i, left=y_i, height=256, width=256)
        img_f = img_f.unsqueeze(0)
        img_g = img_g.unsqueeze(0)  # 2x256x256
        img_all = torch.cat((img_f, img_g), dim=0)
        return img_all, self.target[index]


class MriTrainUnconDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        self.target = torch.empty((0,))
        self.map_organ = {'BRAIN': 1, 'HEAD': 1, 'KNEE': 2, 'CSPINE': 3, 'LSPINE': 4, 'TSPINE': 5,
                          'SPINE': 6, 'CAROTID': 7, 'SHOULDER': 8, 'UTERUS': 9, 'lp': 10,
                          'EMPTY': 0}
        self.uii_acc_map = {'FULL': 8, 'ACS2': 2, 'ACS3': 3, 'ACS5': 5, 'ACS7': 7,
                            'PPA2': 2, 'PPA3': 3}

        tot = 0
        data_root = data_path + '/15T_GE_MR'

        date_folders = os.listdir(data_root)
        for date_folder in date_folders:  # root folder
            patient_folders = os.listdir(os.path.join(data_root, date_folder))
            for patient_folder in patient_folders:  # patient folder
                organ_folders = os.listdir(os.path.join(data_root, date_folder, patient_folder))
                for organ_folder in organ_folders:  # organ folder
                    seq_folders = os.listdir(os.path.join(data_root, date_folder, patient_folder, organ_folder))
                    for seq_folder in seq_folders:  # seq folder
                        if 'FAST' not in seq_folder:
                            path_partial_gt = os.path.join(
                                data_root, date_folder, patient_folder, organ_folder, seq_folder
                            )
                            slice_names_gt = os.listdir(path_partial_gt)
                            for slice_name in slice_names_gt:
                                target_cur = torch.zeros(11)
                                path_fix = os.path.join(path_partial_gt, slice_name)
                                itk_fix = sitk.ReadImage(path_fix)
                                img_fix = sitk.GetArrayFromImage(itk_fix).astype(np.float32)
                                img_fix = img_fix[0, :, :]
                                py_dcm_fix = pydicom.read_file(path_fix)

                                target_cur[0] = 1
                                for organ_name, value in self.map_organ:
                                    if organ_name in path_fix:
                                        target_cur[1] = value
                                        break

                                ts_fix = torch.from_numpy(img_fix)
                                min_f, max_f = torch.min(ts_fix), torch.max(ts_fix)
                                ts_fix = 2.0 * (ts_fix - min_f) / (max_f - min_f)- 1.0

                                target_cur[8], target_cur[9] = min_f, max_f
                                target_cur[10] = tot

                                tot = tot + 1
                                self.data.append(ts_fix)
                                target_cur = target_cur.unsqueeze(0)
                                self.target = torch.cat((self.target, target_cur), dim=0)
        # print(tot)
        # print("GE finished")

        # Philip
        data_root = data_path + '/15T_Philip_MR'
        date_folders = os.listdir(data_root)
        for date_folder in date_folders:
            patient_folders = os.listdir(os.path.join(data_root, date_folder))
            for patient_folder in patient_folders:
                organ_folders = os.listdir(os.path.join(data_root, date_folder, patient_folder))
                for organ_folder in organ_folders:
                    if os.path.exists(os.path.join(data_root, date_folder, patient_folder, organ_folder)):
                        seq_folders = os.listdir(os.path.join(data_root, date_folder, patient_folder, organ_folder))
                        for seq_folder in seq_folders:
                            if 'FAST' not in seq_folder:
                                path_partial_gt = os.path.join(
                                    data_root, date_folder, patient_folder, organ_folder, seq_folder
                                )
                                slice_names_gt = os.listdir(path_partial_gt)
                                for slice_name in slice_names_gt:
                                    target_cur = torch.zeros(11)
                                    path_fix = os.path.join(path_partial_gt, slice_name)
                                    itk_fix = sitk.ReadImage(path_fix)
                                    img_fix = sitk.GetArrayFromImage(itk_fix).astype(np.float32)
                                    img_fix = img_fix[0, :, :]
                                    py_dcm_fix = pydicom.read_file(path_fix)

                                    target_cur[0] = 2
                                    for organ_name, value in self.map_organ:
                                        if organ_name in path_fix:
                                            target_cur[1] = value
                                            break

                                    ts_fix = torch.from_numpy(img_fix)
                                    min_f, max_f = torch.min(ts_fix), torch.max(ts_fix)
                                    ts_fix = 2.0 * (ts_fix - min_f) / (max_f - min_f)- 1.0

                                    target_cur[8], target_cur[9] = min_f, max_f
                                    target_cur[10] = tot

                                    tot = tot + 1
                                    self.data.append(ts_fix)
                                    target_cur = target_cur.unsqueeze(0)
                                    self.target = torch.cat((self.target, target_cur), dim=0)
        # print(tot)
        # print("Philip finished")

        # Siemens
        data_root = data_path + '/15T_Siemens_MR'
        date_folders = os.listdir(data_root)
        for date_folder in date_folders:  # date_folders:
            organ_folders = os.listdir(os.path.join(data_root, date_folder))
            for organ_folder in organ_folders:
                if os.path.exists(os.path.join(data_root, date_folder, organ_folder)):
                    seq_folders = os.listdir(os.path.join(data_root, date_folder, organ_folder))
                    for seq_folder in seq_folders:
                        if 'aca' not in seq_folder:
                            patient_folders = os.listdir(os.path.join(data_root, date_folder, organ_folder, seq_folder))
                            for patient_folder in patient_folders:
                                path_partial_gt = os.path.join(
                                    data_root, date_folder, organ_folder, seq_folder, patient_folder
                                )
                                slice_names_aca = os.listdir(path_partial_gt)
                                for slice_name in slice_names_aca:
                                    target_cur = torch.zeros(11)
                                    path_fix = os.path.join(path_partial_gt, slice_name)
                                    itk_fix = sitk.ReadImage(path_fix)
                                    img_fix = sitk.GetArrayFromImage(itk_fix).astype(np.float32)
                                    img_fix = img_fix[0, :, :]
                                    py_dcm_fix = pydicom.read_file(path_fix)

                                    target_cur[0] = 3
                                    for organ_name, value in self.map_organ:
                                        if organ_name in path_fix:
                                            target_cur[1] = value
                                            break

                                    ts_fix = torch.from_numpy(img_fix)
                                    min_f, max_f = torch.min(ts_fix), torch.max(ts_fix)
                                    ts_fix = 2.0 * (ts_fix - min_f) / (max_f - min_f)- 1.0

                                    target_cur[8], target_cur[9] = min_f, max_f
                                    target_cur[10] = tot

                                    tot = tot + 1
                                    self.data.append(ts_fix)
                                    target_cur = target_cur.unsqueeze(0)
                                    self.target = torch.cat((self.target, target_cur), dim=0)
        # print(tot)
        # print("Siemens finished")

        # UIH
        data_root = data_path + '/15T_UIH_MR'
        seq_folders = os.listdir(data_root)
        for seq_folder in seq_folders:
            if ('ACS' not in seq_folder) and ('PPA' not in seq_folder) and ('FULL' not in seq_folder):
                seq_folder_gt = seq_folder
                path_partial_gt = os.path.join(data_root, seq_folder_gt)
                slice_names = os.listdir(seq_folder_gt)
                for slice_name in slice_names:
                    target_cur = torch.zeros(11)
                    path_fix = os.path.join(path_partial_gt, slice_name)
                    itk_fix = sitk.ReadImage(path_fix)
                    img_fix = sitk.GetArrayFromImage(itk_fix).astype(np.float32)
                    img_fix = img_fix[0, :, :]

                    target_cur[0] = 4
                    for organ_name, value in self.map_organ:
                        if (organ_name in path_fix) or (organ_name.lower() in path_fix):
                            target_cur[1] = value
                            break

                    for acc_rate_name, value in self.uii_acc_map:
                        if acc_rate_name in path_fix:
                            target_cur[3] = value

                    ts_fix = torch.from_numpy(img_fix)
                    min_f, max_f = torch.min(ts_fix), torch.max(ts_fix)
                    ts_fix = 2.0 * (ts_fix - min_f) / (max_f - min_f)- 1.0

                    target_cur[8], target_cur[9] = min_f, max_f
                    target_cur[10] = tot

                    tot = tot + 1
                    self.data.append(ts_fix)
                    target_cur = target_cur.unsqueeze(0)
                    self.target = torch.cat((self.target, target_cur), dim=0)
        # print(tot)
        # print("UIH finished")

        # print(len(self.data))
        # print(self.target.shape)
        # print(tot)
        torch.cuda.empty_cache()

    def __len__(self):
        return len(self.data)

    def __shape__(self):
        return self.data.shape

    def __getitem__(self, index):
        img_gt = self.data[index]
        padding_y_left = (256 - img_gt.shape[1]) // 2
        padding_y_right = 256 - img_gt.shape[1] - padding_y_left
        padding_x_left = (256 - img_gt.shape[0]) // 2
        padding_x_right = 256 - img_gt.shape[0] - padding_x_left
        img_gt = F.pad(img_gt, (max(0, padding_y_left), max(0, padding_y_right),
                                max(0, padding_x_left), max(0, padding_x_right)), mode="constant", value=-1)
        h, w = img_gt.shape[0], img_gt.shape[1]
        x_i = random.randint(0, h - 256)
        y_i = random.randint(0, w - 256)
        img_g = crop(img_gt, top=x_i, left=y_i, height=256, width=256)
        img_g = img_g.unsqueeze(0)  # 2x256x256
        return img_g, self.target[index]


class MriValidConDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        self.target = torch.empty((0,))
        self.map_organ = {'BRAIN': 1, 'HEAD': 1, 'KNEE': 2, 'CSPINE': 3, 'LSPINE': 4, 'TSPINE': 5,
                          'SPINE': 6, 'CAROTID': 7, 'SHOULDER': 8, 'UTERUS': 9, 'lp': 10,
                          'EMPTY': 0}
        self.uii_acc_map = {'FULL': 8, 'ACS2': 2, 'ACS3': 3, 'ACS5': 5, 'ACS7': 7,
                            'PPA2': 2, 'PPA3': 3}

        tot = 0
        # GE
        data_root = data_path + '/15T_GE_MR'

        date_folders = os.listdir(data_root)
        for date_folder in date_folders:  # root folder
            patient_folders = os.listdir(os.path.join(data_root, date_folder))
            for patient_folder in patient_folders:  # patient folder
                organ_folders = os.listdir(os.path.join(data_root, date_folder, patient_folder))
                for organ_folder in organ_folders:  # organ folder
                    seq_folders = os.listdir(os.path.join(data_root, date_folder, patient_folder, organ_folder))
                    for seq_folder in seq_folders:  # seq folder
                        if 'FAST' in seq_folder:
                            path_partial_aca = os.path.join(
                                data_root, date_folder, patient_folder, organ_folder, seq_folder
                            )

                            seq_folder_comp = seq_folder.split('_')
                            seq_folder_comp_new = []
                            for comp_id in range(len(seq_folder_comp)):
                                if not 'FAST' in seq_folder_comp[comp_id]:
                                    seq_folder_comp_new.append(seq_folder_comp[comp_id])
                            seq_gt_folder = '_'.join(seq_folder_comp_new)
                            path_partial_gt = os.path.join(
                                data_root, date_folder, patient_folder, organ_folder, seq_gt_folder
                            )

                            slice_names_aca = os.listdir(path_partial_aca)
                            for slice_name in slice_names_aca:
                                target_cur = torch.zeros(11)
                                path_mov = os.path.join(path_partial_aca, slice_name)
                                itk_mov = sitk.ReadImage(path_mov)
                                img_mov = sitk.GetArrayFromImage(itk_mov).astype(np.float32)
                                img_mov = img_mov[0, :, :]  # shape: 256x256
                                py_dcm_mov = pydicom.read_file(path_mov)

                                path_fix = os.path.join(path_partial_gt, slice_name)
                                itk_fix = sitk.ReadImage(path_fix)
                                img_fix = sitk.GetArrayFromImage(itk_fix).astype(np.float32)
                                img_fix = img_fix[0, :, :]
                                py_dcm_fix = pydicom.read_file(path_fix)

                                target_cur[0] = 1
                                for organ_name, value in self.map_organ.items():
                                    if organ_name in path_mov:
                                        target_cur[1] = value
                                        break

                                ts_mov, ts_fix = torch.from_numpy(img_mov), torch.from_numpy(img_fix)
                                min_m, max_m = torch.min(ts_mov), torch.max(ts_mov)
                                min_f, max_f = torch.min(ts_fix), torch.max(ts_fix)
                                ts_mov = 2.0 * (ts_mov - min_m) / (max_m - min_m)- 1.0
                                ts_fix = 2.0 * (ts_fix - min_f) / (max_f - min_f)- 1.0

                                target_cur[6], target_cur[7], target_cur[8], target_cur[9] = min_m, max_m, min_f, max_f
                                target_cur[10] = tot

                                # finally
                                tot = tot + 1
                                ts_mov = ts_mov.unsqueeze(0)
                                ts_fix = ts_fix.unsqueeze(0)  # 2x256x256
                                img_all = torch.cat((ts_mov, ts_fix), dim=0)
                                self.data.append(img_all)
                                target_cur = target_cur.unsqueeze(0)
                                self.target = torch.cat((self.target, target_cur), dim=0)
        # print(tot)
        # print("GE finished")

        # Philip
        data_root = data_path + '/15T_Philip_MR'
        date_folders = os.listdir(data_root)
        for date_folder in date_folders:
            patient_folders = os.listdir(os.path.join(data_root, date_folder))
            for patient_folder in patient_folders:
                organ_folders = os.listdir(os.path.join(data_root, date_folder, patient_folder))
                for organ_folder in organ_folders:
                    if os.path.exists(os.path.join(data_root, date_folder, patient_folder, organ_folder)):
                        seq_folders = os.listdir(os.path.join(data_root, date_folder, patient_folder, organ_folder))
                        for seq_folder in seq_folders:
                            if 'FAST' in seq_folder:
                                path_partial_aca = os.path.join(
                                    data_root, date_folder, patient_folder, organ_folder, seq_folder
                                )
                                seq_folder_comp = seq_folder.split('_')
                                seq_folder_comp_new = []
                                for comp_id in range(len(seq_folder_comp)):
                                    if not 'FAST' in seq_folder_comp[comp_id]:
                                        seq_folder_comp_new.append(seq_folder_comp[comp_id])
                                seq_gt_folder = '_'.join(seq_folder_comp_new)
                                path_partial_gt = os.path.join(
                                    data_root, date_folder, patient_folder, organ_folder, seq_gt_folder
                                )

                                slice_names_aca = os.listdir(path_partial_aca)
                                for slice_name in slice_names_aca:
                                    target_cur = torch.zeros(11)
                                    path_mov = os.path.join(path_partial_aca, slice_name)
                                    itk_mov = sitk.ReadImage(path_mov)
                                    img_mov = sitk.GetArrayFromImage(itk_mov).astype(np.float32)
                                    img_mov = img_mov[0, :, :]  # shape: 256x256
                                    py_dcm_mov = pydicom.read_file(path_mov)

                                    path_fix = os.path.join(path_partial_gt, slice_name)
                                    itk_fix = sitk.ReadImage(path_fix)
                                    img_fix = sitk.GetArrayFromImage(itk_fix).astype(np.float32)
                                    img_fix = img_fix[0, :, :]
                                    py_dcm_fix = pydicom.read_file(path_fix)

                                    target_cur[0] = 2
                                    for organ_name, value in self.map_organ.items():
                                        if organ_name in path_mov:
                                            target_cur[1] = value
                                            break

                                    tag = (0x0018, 0x0083)
                                    if tag in py_dcm_fix:
                                        if py_dcm_fix[tag].value is not None:
                                            target_cur[5] = float(py_dcm_fix[tag].value)

                                    # 3.归一化
                                    ts_mov, ts_fix = torch.from_numpy(img_mov), torch.from_numpy(img_fix)
                                    min_m, max_m = torch.min(ts_mov), torch.max(ts_mov)
                                    min_f, max_f = torch.min(ts_fix), torch.max(ts_fix)
                                    ts_mov = 2.0 * (ts_mov - min_m) / (max_m - min_m)- 1.0
                                    ts_fix = 2.0 * (ts_fix - min_f) / (max_f - min_f)- 1.0

                                    target_cur[6], target_cur[7], target_cur[8], target_cur[9] = min_m, max_m, min_f, max_f
                                    target_cur[10] = tot

                                    # finally
                                    tot = tot + 1
                                    ts_mov = ts_mov.unsqueeze(0)
                                    ts_fix = ts_fix.unsqueeze(0)  # 2x256x256
                                    img_all = torch.cat((ts_mov, ts_fix), dim=0)
                                    self.data.append(img_all)
                                    target_cur = target_cur.unsqueeze(0)
                                    self.target = torch.cat((self.target, target_cur), dim=0)
        # print(tot)
        # print("Philip finished")

        # Siemens
        data_root = data_path + '/15T_Siemens_MR'
        date_folders = os.listdir(data_root)
        for date_folder in date_folders:  # date_folders:
            organ_folders = os.listdir(os.path.join(data_root, date_folder))
            for organ_folder in organ_folders:
                if os.path.exists(os.path.join(data_root, date_folder, organ_folder)):
                    seq_folders = os.listdir(os.path.join(data_root, date_folder, organ_folder))
                    for seq_folder in seq_folders:
                        if 'aca' in seq_folder:
                            patient_folders = os.listdir(os.path.join(data_root, date_folder, organ_folder, seq_folder))
                            for patient_folder in patient_folders:
                                path_partial_aca = os.path.join(
                                    data_root, date_folder, organ_folder, seq_folder, patient_folder
                                )
                                seq_gt_folder, path_partial_gt = '', ''
                                if '_' in date_folder:
                                    seq_gt_folder = 'GT2_' + seq_folder[5:]
                                    path_partial_gt = os.path.join(
                                        data_root, date_folder, organ_folder, seq_gt_folder, patient_folder
                                    )
                                    if not os.path.exists(path_partial_gt):
                                        seq_gt_folder = 'GT1_' + seq_folder[5:]
                                        path_partial_gt = os.path.join(
                                            data_root, date_folder, organ_folder, seq_gt_folder, patient_folder
                                        )
                                else:
                                    seq_folder_comp = seq_folder.split('_')
                                    seq_folder_comp_new = []
                                    for comp_id in range(len(seq_folder_comp)):
                                        if (not 'aca' in seq_folder_comp[comp_id]) and (
                                        not 'ppa' in seq_folder_comp[comp_id]) and (
                                        not 'av' in seq_folder_comp[comp_id]):
                                            seq_folder_comp_new.append(seq_folder_comp[comp_id])
                                    seq_gt_folder = '_'.join(seq_folder_comp_new)
                                    path_partial_gt = os.path.join(
                                        data_root, date_folder, organ_folder, seq_gt_folder, patient_folder
                                    )

                                slice_names_aca = os.listdir(path_partial_aca)
                                for slice_name in slice_names_aca:
                                    target_cur = torch.zeros(11)
                                    path_mov = os.path.join(path_partial_aca, slice_name)
                                    itk_mov = sitk.ReadImage(path_mov)
                                    img_mov = sitk.GetArrayFromImage(itk_mov).astype(np.float32)
                                    img_mov = img_mov[0, :, :]  # shape: 256x256
                                    py_dcm_mov = pydicom.read_file(path_mov)

                                    path_fix = os.path.join(path_partial_gt, slice_name)
                                    itk_fix = sitk.ReadImage(path_fix)
                                    img_fix = sitk.GetArrayFromImage(itk_fix).astype(np.float32)
                                    img_fix = img_fix[0, :, :]
                                    py_dcm_fix = pydicom.read_file(path_fix)

                                    target_cur[0] = 3
                                    for organ_name, value in self.map_organ.items():
                                        if organ_name in path_mov:
                                            target_cur[1] = value
                                            break

                                    tag = [(0x5200, 0x9229), (0x0018, 0x9115), (0x0018, 0x9069)]
                                    tag_cur = tag[0]
                                    acc__value = 0
                                    dcm_dataset = py_dcm_fix
                                    for i in range(3):
                                        if tag[i] not in dcm_dataset:
                                            break
                                        if i == 2:
                                            dcm_dataset = dcm_dataset[tag[i]].value
                                        else:
                                            dcm_dataset = dcm_dataset[tag[i]]
                                            dcm_dataset = dcm_dataset[0]
                                    if acc__value != 0:
                                        target_cur[3] = float(dcm_dataset)

                                    tag = (0x0018, 0x9073)
                                    if tag in py_dcm_fix:
                                        if py_dcm_fix[tag].value is not None:
                                            target_cur[4] = float(py_dcm_fix[tag].value)

                                    ts_mov, ts_fix = torch.from_numpy(img_mov), torch.from_numpy(img_fix)
                                    min_m, max_m = torch.min(ts_mov), torch.max(ts_mov)
                                    min_f, max_f = torch.min(ts_fix), torch.max(ts_fix)
                                    ts_mov = 2.0 * (ts_mov - min_m) / (max_m - min_m)- 1.0
                                    ts_fix = 2.0 * (ts_fix - min_f) / (max_f - min_f)- 1.0

                                    target_cur[6], target_cur[7], target_cur[8], target_cur[9] = min_m, max_m, min_f, max_f
                                    target_cur[10] = tot

                                    tot = tot + 1
                                    ts_mov = ts_mov.unsqueeze(0)
                                    ts_fix = ts_fix.unsqueeze(0)  # 2x256x256
                                    img_all = torch.cat((ts_mov, ts_fix), dim=0)
                                    self.data.append(img_all)
                                    target_cur = target_cur.unsqueeze(0)
                                    self.target = torch.cat((self.target, target_cur), dim=0)
        # print(tot)
        # print("Siemens finished")
        torch.cuda.empty_cache()

    def __len__(self):
        return len(self.data)

    def __shape__(self):
        return self.data.shape

    def __getitem__(self, index):
        x_fast = self.data[index][0]
        x_GT = self.data[index][1]

        padding_size_x = x_fast.shape[0]
        if padding_size_x % 16 > 0:
            padding_size_x = padding_size_x - padding_size_x % 16 + 16
        padding_size_y = x_fast.shape[1]
        if padding_size_y % 16 > 0:
            padding_size_y = padding_size_y - padding_size_y % 16 + 16
        x_fast = F.pad(x_fast, (
            max(0, padding_size_y - x_fast.shape[1] - (padding_size_y - x_fast.shape[1]) // 2),
            max(0, (padding_size_y - x_fast.shape[1]) // 2),
            max(0, padding_size_x - x_fast.shape[0] - (padding_size_x - x_fast.shape[0]) // 2),
            max(0, (padding_size_x - x_fast.shape[0]) // 2)), mode="constant", value=-1)
        x_GT = F.pad(x_GT, (
            max(0, padding_size_y - x_GT.shape[1] - (padding_size_y - x_GT.shape[1]) // 2),
            max(0, (padding_size_y - x_GT.shape[1]) // 2),
            max(0, padding_size_x - x_GT.shape[0] - (padding_size_x - x_GT.shape[0]) // 2),
            max(0, (padding_size_x - x_GT.shape[0]) // 2)), mode="constant", value=-1)

        x_fast = x_fast.unsqueeze(0)
        x_GT = x_GT.unsqueeze(0)  # 2x256x256
        img_all = torch.cat((x_fast, x_GT), dim=0)
        return img_all, self.target[index]