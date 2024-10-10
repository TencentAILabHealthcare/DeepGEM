def build_dataset(image_set, args):
    dataset_name = args.DATASET.DATASET_NAME

    if dataset_name in ("dataset_deepgem"):
        from .dataset_deepgem import (
            build as build_dataset_deepgem,
        )

        return build_dataset_deepgem(image_set, args)

    raise ValueError(f"dataset {dataset_name} not supported")
