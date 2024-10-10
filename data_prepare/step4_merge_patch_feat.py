import inspect
import os
import sys

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import argparse
import os.path as osp
import pickle
from glob import glob
from multiprocessing import Pool

from rich import print
from rich.progress import track

from configs import get_cfg_defaults, update_default_cfg

"""
sotre all extracted features of patchs into a ".pkl" file
"""
def merge_wsi_feat(wsi_feat_dir) -> None:
    """
       Args:
        wsi_feat_dir: folder path to store the path features
    Returns:
    """

    # obtain all file paths
    files = glob(osp.join(wsi_feat_dir, "*.pkl"))
    save_obj = []
    for fp in files:
        try:
            with open(fp, "rb") as infile:
                obj = pickle.load(infile)
            # add patch name
            obj["feat_name"] = osp.basename(fp).rsplit(".", 1)[0]
            save_obj.append(obj)
        except Exception as e:
            print(f"Error in {fp} as {e}")
            continue

    bname = osp.basename(wsi_feat_dir).lower()  # wsi id
    save_fp = osp.join(merge_feat_save_dir, f"{bname}.pkl")
    with open(save_fp, "wb") as outfile:
        pickle.dump(save_obj, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSI patch features extraction")
    parser.add_argument(
        "--cfg",
        default="./configs/sample.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--workers",
        default=80,
        help="number of the workers",
        type=int,
    )
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    update_default_cfg(cfg)

    # extracted feature storage path
    feat_save_dirx20 = cfg.PRETRAIN.SAVE_DIR
    # output path
    merge_feat_save_dirx20 = cfg.PRETRAIN.SAVE_DIR_COMBINED

    for feat_save_dir, merge_feat_save_dir in zip(
        [feat_save_dirx20], [merge_feat_save_dirx20]
    ):
        print(f"Save to {merge_feat_save_dir}")
        os.makedirs(merge_feat_save_dir, exist_ok=True)
        wsi_dirs = glob(osp.join(feat_save_dir, "*"))

        with Pool(args.workers) as p:
            for _ in track(
                p.imap_unordered(merge_wsi_feat, wsi_dirs), total=len(wsi_dirs)
            ):
                pass