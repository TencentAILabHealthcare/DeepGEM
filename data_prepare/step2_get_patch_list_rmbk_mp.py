### TCGA
import argparse
import glob
import multiprocessing as mp
import pickle
import time
import warnings
from glob import glob
from os import listdir

import cv2
import numpy as np
from PIL import Image

warnings.simplefilter("ignore")


maxdiff = 10
backpercent = 50


def isbackground(img):
    c0 = np.mean(img[:, :, 0])
    c1 = np.mean(img[:, :, 1])
    c2 = np.mean(img[:, :, 2])

    temp = [np.abs(c0 - c1), np.abs(c0 - c2), np.abs(c1 - c2)]

    diff = np.max(temp)

    if diff > maxdiff:
        return False
    else:
        return True


def isbackground2(img):
    img = Image.fromarray(img)
    gray = img.convert("L")
    bw = gray.point(lambda x: 0 if x < 220 else 1, "F")
    avgBkg = np.average(bw)

    if avgBkg >= (backpercent / 100):
        return True
    else:
        return False


def bg(img, a):
    try:
        pic = cv2.imread(img)
        result = isbackground(pic)
        return img, result
    except:
        print("error", img)
        return img, False


def remove_background(img_list):
    new_img_list = []
    for img in img_list:
        try:
            pic = cv2.imread(img)
            result = isbackground(pic) and isbackground2(pic)
            if not result:
                new_img_list.append(img)

        except TypeError as e:
            print(e)
    return new_img_list


def remove_background_mp(img_list):
    new_img_list = []

    start = time.time()
    pool = mp.Pool(mp.cpu_count())
    print("number of cpus: ", mp.cpu_count())

    results = pool.map_async(parallel_worker, [(i, None) for i in img_list])
    remaining = results._number_left

    while True:
        if results.ready():
            break
        remaining = results._number_left
        time.sleep(1)
    results = results.get()
    pool.close()

    for img, re in results:
        if not re:
            new_img_list.append(img)

    end = time.time()

    print("time cost", end - start)
    return new_img_list


def parallel_worker(x):
    return bg(*x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop the WSIs into patches")
    parser.add_argument(
        "--dataset",
        type=str,
        default="./sample_data/patch",
        help="Dataset folder name",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./sample_data",
        help="Output folder name",
    )
    args = parser.parse_args()

    # patches list
    patch_dir = args.dataset
    # output for pkl
    out_dir = args.output

    dict_name2img = dict()
    case_list = sorted(listdir(patch_dir))
    for case in case_list:
        patch_list = sorted(glob(f"{patch_dir}/{case}/*.png"))
        new_patch_list = remove_background_mp(patch_list)
        dict_name2img[case] = new_patch_list
    pickle.dump(dict_name2img, open(f"{out_dir}/dict_name2imgs_path_rmbg.pkl", "wb"))
    print(f"saving dict_name2imgs.pkl to {out_dir}..")
