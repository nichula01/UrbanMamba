import math
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
# from scipy import misc
import torch
import torchvision


MEAN_RGB = [123.675, 116.28, 103.53]
STD_RGB = [58.395, 57.12, 57.375]


def normalize_img(img, mean=MEAN_RGB, std=STD_RGB):
    """Normalize image by subtracting mean and dividing by std."""
    img_array = np.asarray(img, dtype=np.float32)
    normalized_img = np.empty_like(img_array, np.float32)

    for i in range(3):  # Loop over color channels
        normalized_img[..., i] = (img_array[..., i] - mean[i]) / std[i]

    return normalized_img


def _random_resized_params(height, width, scale=(0.5, 2.0), ratio=(3.0 / 4.0, 4.0 / 3.0), max_attempts=10):
    area = height * width
    for _ in range(max_attempts):
        target_area = random.uniform(*scale) * area
        aspect_ratio = math.exp(random.uniform(math.log(ratio[0]), math.log(ratio[1])))

        new_w = int(round(math.sqrt(target_area * aspect_ratio)))
        new_h = int(round(math.sqrt(target_area / aspect_ratio)))

        if 0 < new_w <= width and 0 < new_h <= height:
            top = 0 if height == new_h else random.randint(0, height - new_h)
            left = 0 if width == new_w else random.randint(0, width - new_w)
            return top, left, new_h, new_w

    # Fallback to center crop if sampling failed
    in_short = min(height, width)
    top = (height - in_short) // 2
    left = (width - in_short) // 2
    return top, left, in_short, in_short


def random_resized_crop_seg(img, label, crop_size, scale=(0.5, 2.0), ratio=(0.75, 1.3333333333), ignore_index=255):
    """
    Segmentation-safe RandomResizedCrop for numpy arrays.
    - img: HWC image array
    - label: HW label array
    - crop_size: (h, w)
    """
    h, w = img.shape[:2]
    top, left, new_h, new_w = _random_resized_params(h, w, scale=scale, ratio=ratio)

    img_cropped = img[top:top + new_h, left:left + new_w]
    label_cropped = label[top:top + new_h, left:left + new_w]

    target_h, target_w = crop_size
    img_pil = Image.fromarray(img_cropped.astype(np.uint8))
    label_pil = Image.fromarray(label_cropped.astype(np.uint8), mode='L')

    img_resized = img_pil.resize((target_w, target_h), Image.BILINEAR)
    label_resized = label_pil.resize((target_w, target_h), Image.NEAREST)

    label_resized = np.asarray(label_resized, dtype=label.dtype)
    label_resized[label_resized == 255] = ignore_index

    return np.asarray(img_resized, dtype=img.dtype), label_resized


def class_balanced_random_crop(img, label, crop_size, mean_rgb=MEAN_RGB, ignore_index=255):
    """Pad to crop_size then sample a region that is not dominated by ignore_index."""
    h, w = label.shape
    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_img = np.zeros((H, W, 3), dtype=np.float32)
    pad_label = np.ones((H, W), dtype=np.float32) * ignore_index

    pad_img[:, :, 0] = mean_rgb[0]
    pad_img[:, :, 1] = mean_rgb[1]
    pad_img[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_img[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = img
    pad_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label

    def get_random_cropbox(cat_max_ratio=0.75):
        for _ in range(10):
            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end

    H_start, H_end, W_start, W_end = get_random_cropbox()
    img = pad_img[H_start:H_end, W_start:W_end, :]
    label = pad_label[H_start:H_end, W_start:W_end]
    return img, label


def color_jitter(img, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05):
    """Apply color jitter on numpy image."""
    img_pil = Image.fromarray(img.astype(np.uint8))
    if brightness > 0:
        factor = 1 + random.uniform(-brightness, brightness)
        img_pil = ImageEnhance.Brightness(img_pil).enhance(factor)
    if contrast > 0:
        factor = 1 + random.uniform(-contrast, contrast)
        img_pil = ImageEnhance.Contrast(img_pil).enhance(factor)
    if saturation > 0:
        factor = 1 + random.uniform(-saturation, saturation)
        img_pil = ImageEnhance.Color(img_pil).enhance(factor)
    if hue > 0:
        hsv = np.array(img_pil.convert('HSV'), dtype=np.uint8)
        delta = int(random.uniform(-hue, hue) * 255)
        hsv[..., 0] = (hsv[..., 0].astype(int) + delta) % 255
        img_pil = Image.fromarray(hsv, mode='HSV').convert('RGB')
    return np.array(img_pil, dtype=np.float32)


def gamma_jitter(img, gamma_range=(0.8, 1.2)):
    gamma = random.uniform(*gamma_range)
    img = img.astype(np.float32) / 255.0
    img = np.power(img, gamma)
    img = np.clip(img * 255.0, 0, 255)
    return img


def gaussian_noise(img, max_std=5.0):
    std = random.uniform(0, max_std)
    noise = np.random.normal(0, std, img.shape)
    img = img.astype(np.float32) + noise
    return np.clip(img, 0, 255)


def gaussian_blur(img, prob=0.0, radius_choices=(1, 2)):
    if random.random() > prob or prob <= 0:
        return img
    radius = random.choice(radius_choices)
    img_pil = Image.fromarray(img.astype(np.uint8))
    img_blurred = img_pil.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(img_blurred, dtype=np.float32)


def small_angle_rotate(img, label, max_angle=15.0, crop_size=None, mean_rgb=MEAN_RGB, ignore_index=255):
    angle = random.uniform(-max_angle, max_angle)
    img_pil = Image.fromarray(img.astype(np.uint8))
    label_pil = Image.fromarray(label.astype(np.uint8), mode='L')

    img_rot = img_pil.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=tuple(int(v) for v in mean_rgb))
    label_rot = label_pil.rotate(angle, resample=Image.NEAREST, expand=True, fillcolor=ignore_index)

    img_rot = np.array(img_rot, dtype=np.float32)
    label_rot = np.array(label_rot, dtype=label.dtype)

    if crop_size is None:
        return img_rot, label_rot

    target_h, target_w = crop_size
    h, w = label_rot.shape
    top = max((h - target_h) // 2, 0)
    left = max((w - target_w) // 2, 0)
    bottom = top + target_h
    right = left + target_w

    img_crop = np.zeros((target_h, target_w, 3), dtype=np.float32)
    img_crop[:, :, 0] = mean_rgb[0]
    img_crop[:, :, 1] = mean_rgb[1]
    img_crop[:, :, 2] = mean_rgb[2]
    label_crop = np.ones((target_h, target_w), dtype=label.dtype) * ignore_index

    img_slice = img_rot[top:bottom, left:right]
    label_slice = label_rot[top:bottom, left:right]

    img_crop[:img_slice.shape[0], :img_slice.shape[1]] = img_slice
    label_crop[:label_slice.shape[0], :label_slice.shape[1]] = label_slice

    return img_crop, label_crop

def random_fliplr(pre_img, post_img, label):
    if random.random() > 0.5:
        label = np.fliplr(label)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)

    return pre_img, post_img, label

def random_fliplr_bda(pre_img, post_img, label_1, label_2):
    if random.random() > 0.5:
        label_1 = np.fliplr(label_1)
        label_2 = np.fliplr(label_2)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)

    return pre_img, post_img, label_1, label_2


def random_fliplr_mcd(pre_img, post_img, label_cd, label_1, label_2):
    if random.random() > 0.5:
        label_cd = np.fliplr(label_cd)
        label_1 = np.fliplr(label_1)
        label_2 = np.fliplr(label_2)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)

    return pre_img, post_img, label_cd, label_1, label_2

def random_flipud(pre_img, post_img, label):
    if random.random() > 0.5:
        label = np.flipud(label)
        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)

    return pre_img, post_img, label

def random_flipud_bda(pre_img, post_img, label_1, label_2):
    if random.random() > 0.5:
        label_1 = np.flipud(label_1)
        label_2 = np.flipud(label_2)

        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)

    return pre_img, post_img, label_1, label_2


def random_flipud_mcd(pre_img, post_img, label_cd, label_1, label_2):
    if random.random() > 0.5:
        label_cd = np.flipud(label_cd)
        label_1 = np.flipud(label_1)
        label_2 = np.flipud(label_2)

        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)

    return pre_img, post_img, label_cd, label_1, label_2


def random_rot(pre_img, post_img, label):
    k = random.randrange(3) + 1

    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label = np.rot90(label, k).copy()

    return pre_img, post_img, label


def random_rot_bda(pre_img, post_img, label_1, label_2):
    k = random.randrange(3) + 1

    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label_1 = np.rot90(label_1, k).copy()
    label_2 = np.rot90(label_2, k).copy()

    return pre_img, post_img, label_1, label_2


def random_rot_mcd(pre_img, post_img, label_cd, label_1, label_2):
    k = random.randrange(3) + 1
    
    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label_1 = np.rot90(label_1, k).copy()
    label_2 = np.rot90(label_2, k).copy()
    label_cd = np.rot90(label_cd, k).copy()

    return pre_img, post_img, label_cd, label_1, label_2


def random_crop(img, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w, _ = img.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_image = np.zeros((H, W, 3), dtype=np.float32)

    pad_image[:, :, 0] = mean_rgb[0]
    pad_image[:, :, 1] = mean_rgb[1]
    pad_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pad_image

    def get_random_cropbox(cat_max_ratio=0.75):

        for i in range(10):

            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_image[H_start:H_end, W_start:W_end, 0]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)

    img = pad_image[H_start:H_end, W_start:W_end, :]

    return img


def random_bi_image_crop(pre_img, object, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = object.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    H_start = random.randrange(0, H - crop_size + 1, 1)
    H_end = H_start + crop_size
    W_start = random.randrange(0, W - crop_size + 1, 1)
    W_end = W_start + crop_size

    # H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)

    pre_img = pre_img[H_start:H_end, W_start:W_end, :]
    # post_img = post_img[H_start:H_end, W_start:W_end, :]
    object = object[H_start:H_end, W_start:W_end]
    # cmap = colormap()
    # misc.imsave('cropimg.png',image/255)
    # misc.imsave('croplabel.png',encode_cmap(GT))
    return pre_img, object


def random_crop_new(pre_img, post_img, label, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = label.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_pre_image = np.zeros((H, W, 3), dtype=np.float32)

    pad_post_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_label = np.ones((H, W), dtype=np.float32) * ignore_index

    # pad_pre_image[:, :] = mean_rgb[0]
    pad_pre_image[:, :, 0] = mean_rgb[0]
    pad_pre_image[:, :, 1] = mean_rgb[1]
    pad_pre_image[:, :, 2] = mean_rgb[2]

    pad_post_image[:, :, 0] = mean_rgb[0]
    pad_post_image[:, :, 1] = mean_rgb[1]
    pad_post_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img
    pad_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label

    def get_random_cropbox(cat_max_ratio=0.75):

        for i in range(10):

            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    label = pad_label[H_start:H_end, W_start:W_end]
   
    return pre_img, post_img, label


def random_crop_bda(pre_img, post_img, loc_label, clf_label, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = loc_label.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_pre_image = np.zeros((H, W, 3), dtype=np.float32)

    pad_post_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_loc_label = np.ones((H, W), dtype=np.float32) * ignore_index
    pad_clf_label = np.ones((H, W), dtype=np.float32) * ignore_index

    # pad_pre_image[:, :] = mean_rgb[0]
    pad_pre_image[:, :, 0] = mean_rgb[0]
    pad_pre_image[:, :, 1] = mean_rgb[1]
    pad_pre_image[:, :, 2] = mean_rgb[2]

    pad_post_image[:, :, 0] = mean_rgb[0]
    pad_post_image[:, :, 1] = mean_rgb[1]
    pad_post_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img
    pad_loc_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = loc_label
    pad_clf_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = clf_label

    def get_random_cropbox(cat_max_ratio=0.75):

        for i in range(10):

            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_loc_label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    loc_label = pad_loc_label[H_start:H_end, W_start:W_end]
    clf_label = pad_clf_label[H_start:H_end, W_start:W_end]

    return pre_img, post_img, loc_label, clf_label


def random_crop_mcd(pre_img, post_img, label_cd, label_1, label_2, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = label_1.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_pre_image = np.zeros((H, W, 3), dtype=np.float32)

    pad_post_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_label_cd = np.ones((H, W), dtype=np.float32) * ignore_index
    pad_label_1 = np.ones((H, W), dtype=np.float32) * ignore_index
    pad_label_2 = np.ones((H, W), dtype=np.float32) * ignore_index

    # pad_pre_image[:, :] = mean_rgb[0]
    pad_pre_image[:, :, 0] = mean_rgb[0]
    pad_pre_image[:, :, 1] = mean_rgb[1]
    pad_pre_image[:, :, 2] = mean_rgb[2]

    pad_post_image[:, :, 0] = mean_rgb[0]
    pad_post_image[:, :, 1] = mean_rgb[1]
    pad_post_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img

    pad_label_cd[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label_cd
    pad_label_1[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label_1
    pad_label_2[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label_2

    def get_random_cropbox(cat_max_ratio=0.75):

        for i in range(10):

            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_label_1[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    label_cd = pad_label_cd[H_start:H_end, W_start:W_end]
    label_1 = pad_label_1[H_start:H_end, W_start:W_end]
    label_2 = pad_label_2[H_start:H_end, W_start:W_end]

    return pre_img, post_img, label_cd, label_1, label_2
