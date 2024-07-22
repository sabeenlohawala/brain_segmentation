import pickle
from collections import Counter

import numpy as np
from utils import load_brains, mapping

file_path = (
    "/om2/user/matth406/nobrainer_data/data/SharedData/segmentation/freesurfer_asegs"
)
idx_path = "/om2/user/matth406/nobrainer_data/data/SharedData/segmentation/idx.dat"

with open(idx_path, "rb") as f:
    train_images, train_masks, _, _, _, _ = pickle.load(f)

end_idx = 1000

train_images = train_images[: int(end_idx * 0.8)]
train_masks = train_masks[: int(end_idx * 0.8)]

height, width = 162, 194
too_small = 0

for activate_mapping in [True, False]:
    pixel_counts_no_exclude = Counter(dict())
    pixel_counts_exclude = Counter(dict())
    pixel_counts_crop = Counter(dict())
    pixel_counts_small = Counter(dict())
    pixel_counts_padding = Counter(dict())
    pixel_counts_orig = {i: 0 for i in range(107)}
    for image_file, mask_file in zip(train_images, train_masks):
        brain, brain_mask, _ = load_brains(image_file, mask_file, file_path)
        if activate_mapping:
            brain_mask = mapping(brain_mask)

        # slice the MRI volume in 3 directions
        for d in range(3):
            for i in range(brain_mask.shape[d]):
                # get the slice
                if d == 0:
                    brain_slice = brain[i, :, :]
                    mask_slice = brain_mask[i, :, :]
                elif d == 1:
                    brain_slice = brain[:, i, :]
                    mask_slice = brain_mask[:, i, :]
                elif d == 2:
                    brain_slice = brain[:, :, i]
                    mask_slice = brain_mask[:, :, i]

                unique, counts = np.unique(mask_slice, return_counts=True)
                image_dict = {i: j for i, j in zip(unique, counts)}
                image_dict = Counter(image_dict)

                pixel_counts_no_exclude = pixel_counts_no_exclude + image_dict

                # skip slices with no or little brain (20% cutoff)
                if np.sum(brain_slice) < 52428:
                    continue

                pixel_counts_exclude = pixel_counts_exclude + image_dict

                # find the crop
                for j in range(256):
                    if (brain_slice[j] != 0).any():
                        cut_top_temp = j
                        break
                for j in range(256):
                    if (brain_slice[255 - j] != 0).any():
                        cut_bottom_temp = 255 - j
                        break
                for j in range(256):
                    if (brain_slice[:, j] != 0).any():
                        cut_left_temp = j
                        break
                for j in range(256):
                    if (brain_slice[:, 255 - j] != 0).any():
                        cut_right_temp = 255 - j
                        break

                height_temp = cut_bottom_temp - cut_top_temp + 1
                width_temp = cut_right_temp - cut_left_temp + 1

                assert height_temp <= height, "Crop height is too big"
                assert width_temp <= width, "Crop width is too big"

                # crop image-optimal patch:
                brain_slice = brain_slice[
                    cut_top_temp : cut_bottom_temp + 1,
                    cut_left_temp : cut_right_temp + 1,
                ]
                mask_slice = mask_slice[
                    cut_top_temp : cut_bottom_temp + 1,
                    cut_left_temp : cut_right_temp + 1,
                ]

                unique, counts = np.unique(mask_slice, return_counts=True)
                image_dict = {i: j for i, j in zip(unique, counts)}
                image_dict = Counter(image_dict)
                pixel_counts_crop = pixel_counts_crop + image_dict

                if brain_slice.shape[0] < 50 or brain_slice.shape[1] < 50:
                    too_small += 1
                    continue

                pixel_counts_small = pixel_counts_small + image_dict

                # adjust the crop to largest rectangle
                if height_temp < height:
                    diff = height - height_temp
                    # even difference
                    if (diff % 2) == 0:
                        # cut_bottom_temp += diff//2
                        # cut_top_temp -= diff//2
                        brain_slice = np.pad(
                            brain_slice, ((diff // 2, diff // 2), (0, 0))
                        )
                        mask_slice = np.pad(
                            mask_slice, ((diff // 2, diff // 2), (0, 0))
                        )
                    # odd difference
                    else:
                        # cut_bottom_temp += diff//2
                        # cut_top_temp -= diff//2 + 1
                        brain_slice = np.pad(
                            brain_slice, ((diff // 2, diff // 2 + 1), (0, 0))
                        )
                        mask_slice = np.pad(
                            mask_slice, ((diff // 2, diff // 2 + 1), (0, 0))
                        )
                if width_temp < width:
                    diff = width - width_temp
                    # even difference
                    if (diff % 2) == 0:
                        # cut_right_temp += diff//2
                        # cut_left_temp -= diff//2
                        brain_slice = np.pad(
                            brain_slice, ((0, 0), (diff // 2, diff // 2))
                        )
                        mask_slice = np.pad(
                            mask_slice, ((0, 0), (diff // 2, diff // 2))
                        )
                    # odd difference
                    else:
                        # cut_right_temp += diff//2
                        # cut_left_temp -= diff//2 + 1
                        brain_slice = np.pad(
                            brain_slice, ((0, 0), (diff // 2, diff // 2 + 1))
                        )
                        mask_slice = np.pad(
                            mask_slice, ((0, 0), (diff // 2, diff // 2 + 1))
                        )

                unique, counts = np.unique(mask_slice, return_counts=True)
                image_dict = {i: j for i, j in zip(unique, counts)}
                image_dict = Counter(image_dict)
                pixel_counts_padding = pixel_counts_padding + image_dict

                if activate_mapping:
                    for i, j in zip(unique, counts):
                        pixel_counts_orig[i] += j

    pixel_counts_no_exclude = dict(pixel_counts_no_exclude)
    pixel_counts_exclude = dict(pixel_counts_exclude)
    pixel_counts_crop = dict(pixel_counts_crop)
    pixel_counts_small = dict(pixel_counts_small)
    pixel_counts_padding = dict(pixel_counts_padding)

    print(f"With Mapping: {activate_mapping}")
    print("Pixel Counts without Exclude")
    print(pixel_counts_no_exclude)
    print("Pixel Counts with Exclude")
    print(pixel_counts_exclude)
    print("Pixel Counts After Crop")
    print(pixel_counts_crop)
    print("Pixel Counts After remove Small patches")
    print(pixel_counts_small)
    print("Pixel Counts After Padding")
    print(pixel_counts_padding)
    if activate_mapping:
        print("Pixel Counts Original")
        print(pixel_counts_orig)

    with open(
        f"../images/pixel_counts_no_exclude_{activate_mapping}.pkl", "wb"
    ) as pickle_file:
        pickle.dump(pixel_counts_no_exclude, pickle_file)
    with open(
        f"../images/pixel_counts_exclude_{activate_mapping}.pkl", "wb"
    ) as pickle_file:
        pickle.dump(pixel_counts_exclude, pickle_file)
    with open(
        f"../images/pixel_counts_crop_{activate_mapping}.pkl", "wb"
    ) as pickle_file:
        pickle.dump(pixel_counts_crop, pickle_file)
    with open(
        f"../images/pixel_counts_small_{activate_mapping}.pkl", "wb"
    ) as pickle_file:
        pickle.dump(pixel_counts_small, pickle_file)
    with open(
        f"../images/pixel_counts_padding_{activate_mapping}.pkl", "wb"
    ) as pickle_file:
        pickle.dump(pixel_counts_padding, pickle_file)
    with open(
        f"../images/pixel_counts_orig_{activate_mapping}.pkl", "wb"
    ) as pickle_file:
        pickle.dump(pixel_counts_orig, pickle_file)

# To load the data back:
# with open("../images/pixel_count.pkl", 'rb') as pickle_file:
#    loaded_data = pickle.load(pickle_file)
