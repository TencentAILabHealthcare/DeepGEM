import os
import pickle

import numpy as np
import torch
import torch.utils.data as data_utils


class DeepGEMDataset(data_utils.Dataset):
    """
    DeepGEM Dataset
    """
    def __init__(
        self,
        data_info=None,
        num_class=2,
        is_training=False,
        feature_len=500,
    ):
        super().__init__()

        self.is_training = is_training

        self.wsi_loc = []
        for i in range(len(data_info["pid"])):
            pid, feat_fp, label = data_info["pid"][i], data_info["feat_fp"][i], data_info["label"][i]
            self.wsi_loc.append((pid, feat_fp, label))
        sorted(self.wsi_loc, key=lambda x: x[0])

        self.num_class = num_class
        self.feature_len = feature_len
        self.is_training = is_training

    def __len__(self):
        return len(self.wsi_loc)

    def __getitem__(self, index):
        feat_len = self.feature_len
        image_id, patch_feat_fp, label = self.wsi_loc[index]
        label = torch.tensor(label).long()

        # load a wsi
        patch_feat = []
        with open(patch_feat_fp, "rb") as infile:
            patch_feat = pickle.load(infile)

        # train or val
        ret_feat = []
        ret_feat_name = []
        for each_ob in patch_feat:
            if "tr" in each_ob.keys():
                c_feat = each_ob["tr"]
                select_idx = np.random.choice(c_feat.shape[0], 1).item()
                select_feat = c_feat[select_idx]
                select_feat_name = each_ob["feat_name"]
            else:
                select_feat = each_ob["val"]
                select_feat_name = each_ob["feat_name"]

            select_feat = select_feat.reshape(-1)
            ret_feat.append(select_feat)
            ret_feat_name.append(select_feat_name)

        ret_feat = np.stack(ret_feat)

        mask = np.ones(len(ret_feat))

        if len(ret_feat) > feat_len:
            ret_feat = ret_feat[:feat_len]
            mask = mask[:feat_len]
            ret_feat_name = ret_feat_name[:feat_len]

        cur_len = len(ret_feat)
        if cur_len < feat_len:
            residual = feat_len - len(ret_feat)
            if residual == 0:
                pass
            else:
                append_data = np.zeros((residual, ret_feat.shape[1]))
                append_data_name = [ret_feat_name[0]] * residual
                append_mask = np.zeros(residual)

                ret_feat = np.concatenate([ret_feat, append_data])
                mask = np.concatenate([mask, append_mask])
                ret_feat_name = ret_feat_name + append_data_name

        # type setting
        ret_feat = torch.from_numpy(ret_feat).float()
        mask = torch.from_numpy(mask)
        mask = mask.bool()

        label = label.reshape(-1,).repeat(len(mask))

        target = {
            "label": label,
            "pid": image_id,
            "patch_name": ret_feat_name}

        return ret_feat, mask, target


def build(image_set, args):
    print(f"Build {image_set} dataset")
    if args.input_data.split(".")[-1] == "pickle":
        with open(args.input_data, 'rb') as f:
            data_info = pickle.load(f)
        data_info = data_info[args.wsi_type][args.gene]["test"]
        data_info["feat_fp"] = [os.path.join(args.feat_dir, pid + ".pkl") for pid in data_info["pid"]]
    elif args.input_data.split(".")[-1] == "csv":
        with open(args.input_data, 'r', encoding="utf-8") as f:
            lines = f.readlines()
        pid_list, label_list = [], []
        for line in lines[1:]:
            pid_list.append(line.split(",")[0])
            label_list.append(int(line.split(",")[1].split("\n")[0]))
        data_info = {}
        data_info["pid"] = pid_list
        data_info["label"] = label_list
        data_info["feat_fp"] = [os.path.join(args.feat_dir, pid + ".pkl") for pid in data_info["pid"]]

    with open(args.checkpoint, 'rb') as f:
        parameter = pickle.load(f)["parameter"]
    feature_len = parameter['feature_len']

    dataset = DeepGEMDataset(
        data_info=data_info,
        feature_len=feature_len
    )
    return dataset
