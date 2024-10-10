###
import argparse
import glob
import os
import pickle
import threading
import time
import warnings
from os import makedirs, path
from os.path import join

import numpy as np
import openslide as slide
from skimage import img_as_ubyte, io, transform
from skimage.util import img_as_ubyte
from tqdm import tqdm

warnings.simplefilter("ignore")


def parse_filename_from_directory(input_file_list):
    output_file_list = [
        os.path.basename(os.path.splitext(item)[0]) for item in input_file_list
    ]
    return output_file_list


def crop_slide(
    img,
    save_slide_path,
    position=(0, 0),
    step=(0, 0),
    patch_size=224,
    scale=10,
    down_scale=1,
):  # position given as (x, y) at nx scale
    patch_name = "{}_{}".format(step[0], step[1])

    img_nx_path = join(
        save_slide_path,
        f"{patch_name}-tile-r{position[1] * down_scale}-c{position[0] * down_scale}-{patch_size}x{patch_size}.png",
    )
    if path.exists(img_nx_path):
        return 1

    img_x = img.read_region(
        (position[0] * down_scale, position[1] * down_scale),
        0,
        (patch_size * down_scale, patch_size * down_scale),
    )
    img_x = np.array(img_x)[..., :3]

    img = transform.resize(
        img_x,
        (img_x.shape[0] // down_scale, img_x.shape[0] // down_scale),
        order=1,
        anti_aliasing=False,
    )
    try:
        io.imsave(img_nx_path, img_as_ubyte(img))
    except Exception as e:
        print(e)


def slide_to_patch(out_base, img_slides, patch_size, step_size, scale, down_scale=1):
    makedirs(out_base, exist_ok=True)
    for s in tqdm(range(len(img_slides))):
        img_slide = img_slides[s]
        img_name = img_slide.split(path.sep)[-1].split(".")[0]
        bag_path = join(out_base, img_name)

        makedirs(bag_path, exist_ok=True)
        img = slide.OpenSlide(img_slide)

        try:
            if int(np.floor(float(img.properties["openslide.mpp-x"]) * 10)) == 2:
                down_scale = 40 // scale
            else:
                down_scale = 20 // scale
        except Exception as e:
            print("tiff --> No properties 'openslide.mpp-x'")

        dimension = img.level_dimensions[0]
        
        # dimension and step at given scale
        step_y_max = int(np.floor(dimension[1] / (step_size * down_scale)))  # rows
        step_x_max = int(np.floor(dimension[0] / (step_size * down_scale)))  # columns
        print("number :", step_x_max, step_y_max, step_x_max * step_y_max)
        num = step_x_max * step_y_max
        count = 0
        for j in range(step_y_max):
            for i in range(step_x_max):
                start_time = time.time()
                crop_slide(
                    img,
                    bag_path,
                    (i * step_size, j * step_size),
                    step=(j, i),
                    patch_size=patch_size,
                    scale=scale,
                    down_scale=down_scale,
                )
                end_time = time.time()
                count += 1
                print(f"{count}/{num}", (end_time - start_time) / 60)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop the WSIs into patches")
    parser.add_argument(
        "--num_threads",
        type=int,
        default=16,
        help="Number of threads for parallel processing, too large may result in errors",
    )
    parser.add_argument(
        "--overlap", type=int, default=0, help="Overlap pixels between adjacent patches"
    )
    parser.add_argument("--patch_size", type=int, default=1120, help="Patch size")
    parser.add_argument("--scale", type=int, default=20, help="20x 10x 5x")
    parser.add_argument(
        "--dataset", type=str, default="./sample_data/WSI", help="Dataset folder name"
    )
    parser.add_argument(
        "--output", type=str, default="./sample_data/patch", help="Output folder name"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display patch numbers under this setting",
    )
    parser.add_argument(
        "--annotation", action="store_true", help="Obtain patches in annotation region"
    )
    args = parser.parse_args()

    print(
        "Cropping patches, this could take a while for big dataset, please be patient"
    )
    step = args.patch_size - args.overlap

    # obtain dataset paths
    path_base = args.dataset
    out_base = args.output
    if path.isdir(path_base):
        all_slides = (
            glob.glob(f"{path_base}/*.svs")
            + glob.glob(f"{path_base}/*.tif")
            + glob.glob(f"{path_base}/*.tiff")
            + glob.glob(f"{path_base}/*.mrxs")
            + glob.glob(f"{path_base}/*.ndpi")
        )
    else:
        raise ValueError(f"Please check dataset folder {path_base}")

    print("Number of .svs .mrxs .ndpi .tif/f", len(all_slides))

    # crop all patches
    each_thread = int(np.floor(len(all_slides) / args.num_threads))
    threads = []
    for i in range(args.num_threads):
        if i < (args.num_threads - 1):
            t = threading.Thread(
                target=slide_to_patch,
                args=(
                    out_base,
                    all_slides[each_thread * i : each_thread * (i + 1)],
                    args.patch_size,
                    step,
                    args.scale,
                ),
            )
        else:
            t = threading.Thread(
                target=slide_to_patch,
                args=(
                    out_base,
                    all_slides[each_thread * i :],
                    args.patch_size,
                    step,
                    args.scale,
                ),
            )
        threads.append(t)

    for thread in threads:
        thread.start()

    dict_name2img = dict()
    ID_list = parse_filename_from_directory(all_slides)
    print(ID_list)
    for case_ID in ID_list:
        patch_list = glob.glob(f"{out_base}/{case_ID}/*.png")
        dict_name2img[case_ID] = patch_list
    pickle.dump(dict_name2img, open(f"{out_base}/dict_name2imgs.pkl", "wb"))
    print(f"saving dict_name2imgs.pkl to {out_base}..")
