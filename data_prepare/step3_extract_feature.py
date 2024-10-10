"""
Extract feature offline
"""
import inspect
import os
import sys

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import argparse
import multiprocessing
import os.path as osp
import pickle
from concurrent.futures import ThreadPoolExecutor

import albumentations as alb
import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet
from PIL import Image
from rich import progress
from timm.models.layers.helpers import to_2tuple
from torch import nn
from torchvision import transforms

from configs import get_cfg_defaults, update_default_cfg

val_trans = alb.Compose(
    [
        alb.Resize(224, 224),
        alb.Normalize(),
        ToTensorV2(),
    ]
)


class PatchDataset(data_utils.Dataset):
    def __init__(self, img_fp_list, trans):
        print("dataset for imagenet")
        self.img_fp_list = img_fp_list
        self.trans = trans

    def __len__(self):
        return len(self.img_fp_list)

    def __getitem__(self, idx):
        try:
            img_fp = self.img_fp_list[idx]
            img = cv2.imread(img_fp)
            img = img[:, :, ::-1]
        except:
            # if img is None:
            img = np.zeros((224, 224, 3)).astype("uint8")

        pid = osp.basename(osp.dirname(img_fp))
        img_bname = osp.basename(img_fp).rsplit(".", 1)[0]
        aug_img = self.trans(image=img)["image"]
        return pid, img_bname, aug_img


class CTransPathDataset(data_utils.Dataset):
    def __init__(self, img_fp_list):
        super().__init__()
        print("dataset for CTransPath")
        self.img_fp_list = img_fp_list
        self.trans = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def __len__(self):
        return len(self.img_fp_list)

    def __getitem__(self, idx):
        try:
            img_fp = self.img_fp_list[idx]
            img = Image.open(img_fp).convert("RGB")
        except:
            # if img is None:
            img = Image.new("RGB", (256, 256), (0, 0, 0))
        aug_img = self.trans(img)

        pid = osp.basename(osp.dirname(img_fp))
        img_bname = osp.basename(img_fp).rsplit(".", 1)[0]
        return pid, img_bname, aug_img

def save_val_feat_in_thread(batch_pid, batch_img_bname, batch_val_feat, save_dir):
    for b_idx, (pid, img_bname, val_feat) in enumerate(
        zip(batch_pid, batch_img_bname, batch_val_feat)
    ):
        feat_save_dir = osp.join(save_dir, pid)
        os.makedirs(feat_save_dir, exist_ok=True)
        feat_save_name = osp.join(feat_save_dir, f"{img_bname}.pkl")

        if osp.exists(feat_save_name):
            with open(feat_save_name, "rb") as infile:
                save_dict = pickle.load(infile)
        else:
            save_dict = {}

        save_dict["val"] = val_feat

        with open(feat_save_name, "wb") as outfile:
            pickle.dump(save_dict, outfile)


def pred_and_save_with_dataloader(
    model,
    img_fp_list,
    save_dir,
    batch_size=512,
    dataset_class="imagenet",
):
    print("start dataloader..")
    model.eval()

    num_processes = multiprocessing.cpu_count()
    num_processes = min(48, num_processes)

    executor = ThreadPoolExecutor(max_workers=num_processes)

    if dataset_class == "imagenet":
        val_dataset = PatchDataset(img_fp_list, trans=val_trans)
    elif dataset_class == "ctranspath":
        val_dataset = CTransPathDataset(img_fp_list)

    val_dl = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_processes,
        shuffle=False,
        drop_last=False,
    )

    for batch in progress.track(val_dl):
        batch_pid, batch_img_bname, batch_tr_img = batch
        batch_tr_img = batch_tr_img

        with torch.no_grad():
            val_feat = model(batch_tr_img)
            val_feat = val_feat.cpu().numpy()
            executor.submit(
                save_val_feat_in_thread, batch_pid, batch_img_bname, val_feat, save_dir
                )

    print("task finished..")

# pre-download from https://github.com/lukemelas/EfficientNet-PyTorch/releases/tag/1.0

class EffNet(nn.Module):
    def __init__(self, efname="efficientnet-b0", model_path=""):
        super(EffNet, self).__init__()
        self.model = EfficientNet.from_name(efname)
        print(f"Load pretrain model from {model_path}")
        self.model.load_state_dict(torch.load(model_path))

    def forward(self, data):
        bs = data.shape[0]
        feat = self.model.extract_features(data)
        feat = nn.functional.adaptive_avg_pool2d(feat, output_size=(1))
        feat = feat.view(bs, -1)
        return feat

class ConvStem(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(
                nn.Conv2d(
                    input_dim,
                    output_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

def ctranspath(modelname):
    model = timm.create_model(
        modelname, embed_layer=ConvStem, pretrained=False
    )
    return model

class CTransPath(nn.Module):
    def __init__(self, modelname="swin_tiny_patch4_window7_224", model_path=""):
        super(CTransPath, self).__init__()
        self.model = ctranspath(modelname)
        self.model.head = nn.Identity()
        print(f"Load pretrain model from {model_path}")
        self.model.load_state_dict(torch.load(model_path)["model"], strict=True)

    def forward(self, data):
        bs = data.shape[0]
        feat = self.model(data)
        feat = feat.view(bs, -1)
        return feat

def get_model(model_name="", model_path=""):
    if model_name == "efficientnet-b0":
        model = EffNet(efname="efficientnet-b0", model_path=model_path)
    elif model_name == "ctranspath":
        model = CTransPath(modelname="swin_tiny_patch4_window7_224", model_path=model_path)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSI patch features extraction")
    parser.add_argument(
        "--cfg",
        default="./configs/sample.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    update_default_cfg(cfg)

    wsi_patch_info = cfg.PATCH.IDX2IMG
    model_name = cfg.PRETRAIN.MODEL_NAME
    num_classes = cfg.PRETRAIN.NUM_CLASSES
    model_path = cfg.PRETRAIN.MODEL_PATH
    times = cfg.PRETRAIN.TRAIN_FEA_TIMES
    save_dir = cfg.PRETRAIN.SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    # creat model
    model = get_model(
        model_name=model_name, model_path=model_path)

    # load path of original images
    print("load img path..")
    img_fp_list = []
    print(wsi_patch_info)
    count = 0
    filter_image = []
    with open(wsi_patch_info, "rb") as fp:
        dt = pickle.load(fp)
        for k, v in dt.items():
            img_fp_list.extend(v)
            if len(v) > 0:
                count += 1
            else:
                filter_image.append(k)

    img_fp_list = sorted(img_fp_list)
    print("total_count", count)
    print("filter_image", filter_image)
    print(f"Len of img {len(img_fp_list)}")

    if model_name == "ctranspath":
        dataset_class = "ctranspath"
    else:
        dataset_class = "imagenet"

    pred_and_save_with_dataloader(
        model,
        img_fp_list,
        save_dir=save_dir,
        batch_size=4,
        dataset_class=dataset_class,
    )
