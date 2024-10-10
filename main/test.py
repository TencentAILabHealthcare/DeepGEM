# -*- coding: utf-8 -*-
import argparse
import datetime
import os
import os.path as osp
import pickle
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import util.misc as utils
from datasets import build_dataset
from engine import evaluate
from models import build_DeepGEM
from torch.utils.data import DataLoader
from util.logger import getLog

sys.path.insert(0, osp.abspath("./"))
from configs import get_cfg_defaults, update_default_cfg


def main(cfg):
    time_now = datetime.datetime.now()

    unique_comment = f"{cfg.MODEL.MODEL_NAME}{time_now.month}{time_now.day}{time_now.hour}{time_now.minute}"
    cfg.TRAIN.OUTPUT_DIR = osp.join(
        cfg.TRAIN.OUTPUT_DIR, unique_comment
    )
    os.makedirs(cfg.TRAIN.OUTPUT_DIR, exist_ok=True)

    logger = getLog(cfg.TRAIN.OUTPUT_DIR + "/log.txt", screen=True)

    # TODO
    utils.init_distributed_mode(cfg)

    device = torch.device("cuda")

    seed = cfg.TRAIN.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_name = cfg.MODEL.MODEL_NAME
    logger.info(f"Build Model: {model_name}")
    if model_name == "deepgem":
        model, criterion = build_DeepGEM(cfg)
        model.to(device)
    else:
        logger.info(f"Model name not found, {model_name}")
        raise ValueError(f"Model name not found")

    if cfg.TRAIN.LOSS_NAME == "be":
        criterion = nn.CrossEntropyLoss()

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dataset_test = build_dataset(image_set="test", args=cfg)

    print("dataset...")
    if cfg.distributed:
        logger.info("ddp is not implemented")
        raise ModuleNotFoundError(f"ddp is not implemented")
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    collate_func = (
        utils.collate_fn_multi_modal
        if cfg.DATASET.DATASET_NAME in ("vilt", "vilt_surv", "unit")
        else utils.collate_fn
    )


    with open(args.checkpoint, "rb") as f:
        parameter = pickle.load(f)["parameter"]
    cfg.TRAIN.BATCH_SIZE = parameter['batch_size']
    data_loader_test = DataLoader(
        dataset_test,
        int(cfg.TRAIN.BATCH_SIZE * 1.2),
        sampler=sampler_test,
        drop_last=False,
        collate_fn=collate_func,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=True,
    )

    print("loading model........")
    logger.info(f"resume from {cfg.checkpoint}")

    with open(cfg.checkpoint, "rb") as f:
        checkpoint = pickle.load(f)["checkpoint"]

    missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
        checkpoint, strict=False
    )
    unexpected_keys = [
        k
        for k in unexpected_keys
        if not (k.endswith("total_params") or k.endswith("total_ops"))
    ]
    if len(missing_keys) > 0:
        logger.info("Missing Keys: {}".format(missing_keys))
    if len(unexpected_keys) > 0:
        logger.info("Unexpected Keys: {}".format(unexpected_keys))

    test_stats, test_result = evaluate(
        logger,
        model,
        criterion,
        data_loader_test,
        device,
        cfg.distributed,
        display_header="Test",
        kappa_flag=cfg.TRAIN.KAPPA,
    )

    print(test_stats)
    if cfg.save_testfile is not False:
        test_record = {
            "ID": test_result["img_id"],
            "negative": test_result["pred"][:, 0],
            "positive": test_result["pred"][:, 1],
            "label": test_result["label"],
        }
        df = pd.DataFrame(test_record)
        df.to_csv(os.path.join(cfg.TRAIN.OUTPUT_DIR, cfg.gene + "_test_record_" + cfg.wsi_type + ".csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DeepGEM")
    parser.add_argument(
        "--cfg",
        default="./configs/internal.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--input_data",
        default="./data/internal/internal.pickle",
        help="",
        type=str,
    )
    parser.add_argument(
        "--feat_dir",
        default="./data/internal/combined_feat",
        help="",
        type=str,
    )
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/DeepGEM/model_ExcisionalBiopsy_EGFR.pickle",
        help="",
        type=str,
    )
    parser.add_argument(
        "--gene",
        default="EGFR",
        help="Gene Type",
        type=str,
    )
    parser.add_argument(
        "--wsi_type",
        default="ExcisionalBiopsy",
        help="WSI Type: ",
        type=str,
    )
    parser.add_argument(
        "--is_training",
        default=0,
        help="is_training",
        type=int,
    )
    parser.add_argument(
        "--save_testfile",
        default=False,
        help="",
        type=bool,
    )

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    update_default_cfg(cfg)
    cfg.is_training = args.is_training
    cfg.input_data = args.input_data
    cfg.checkpoint = args.checkpoint
    cfg.gene = args.gene
    cfg.wsi_type = args.wsi_type
    cfg.feat_dir = args.feat_dir
    cfg.save_testfile = args.save_testfile

    np.random.seed(666)
    torch.manual_seed(666)
    torch.cuda.manual_seed_all(666)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(cfg)


