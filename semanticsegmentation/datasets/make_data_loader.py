import argparse
import os
import random
from collections import OrderedDict, defaultdict

import imageio
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

import UrbanMamba.semanticsegmentation.datasets.imutils as imutils


def img_loader(path):
    img = np.array(imageio.imread(path), np.float32)
    return img


def one_hot_encoding(image, num_classes=8):
    # Create a one hot encoded tensor
    one_hot = np.eye(num_classes)[image.astype(np.uint8)]

    # Move the channel axis to the front
    # one_hot = np.moveaxis(one_hot, -1, 0)

    return one_hot

COLOR_MAP_LOVEDA = OrderedDict(
    Background=(255, 255, 255),
    Building=(255, 0, 0),
    Road=(255, 255, 0),
    Water=(0, 0, 255),
    Barren=(159, 129, 183),
    Forest=(0, 255, 0),
    Agricultural=(255, 195, 128),
)


LABEL_MAP_LOVEDA = OrderedDict(
    Background=0,
    Building=1,
    Road=2,
    Water=3,
    Barren=4,
    Forest=5,
    Agricultural=6
)


class SemanticDatasetLOVEDA(Dataset):
    """LoveDA semantic segmentation dataset with geometric + photometric augmentations."""

    def __init__(
        self,
        dataset_path,
        data_list,
        crop_size,
        mode='train',
        cfg=None,
        max_iters=None,
        data_loader=img_loader,
        ignore_index=255,
    ):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.mode = mode
        self.ignore_index = ignore_index
        self.cfg = cfg

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]

        if isinstance(crop_size, (list, tuple)):
            self.crop_size = (crop_size[0], crop_size[1])
        else:
            self.crop_size = (crop_size, crop_size)

        self.samples = self._build_samples()
        self.domain_to_indices = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            self.domain_to_indices[sample['domain']].append(idx)

    def _infer_domain(self, path):
        lower = path.lower()
        if 'urban' in lower:
            return 'urban'
        if 'rural' in lower:
            return 'rural'
        return 'unknown'

    def _build_samples(self):
        samples = []
        for entry in self.data_list:
            tokens = entry.split()
            if len(tokens) >= 2:
                img_rel, label_rel = tokens[0], tokens[1]
            else:
                img_rel = tokens[0]
                # try common LoveDA pattern
                if 'images_png' in img_rel:
                    label_rel = img_rel.replace('images_png', 'masks_png')
                elif 'images' in img_rel:
                    label_rel = img_rel.replace('images', 'masks')
                else:
                    label_rel = img_rel

            img_path = os.path.join(self.dataset_path, img_rel)
            label_path = os.path.join(self.dataset_path, label_rel)
            domain = self._infer_domain(img_rel)
            sample_id = os.path.splitext(os.path.basename(img_rel))[0]
            samples.append({'image': img_path, 'label': label_path, 'domain': domain, 'id': sample_id})
        return samples

    def _photometric_augment(self, img, domain=None):
        cfg = self.cfg.AUG if self.cfg is not None else None
        if cfg is not None and cfg.LOVEDA_COLOR_JITTER:
            img = imutils.color_jitter(
                img,
                brightness=cfg.COLOR_JITTER_BCS,
                contrast=cfg.COLOR_JITTER_BCS,
                saturation=cfg.COLOR_JITTER_BCS,
                hue=cfg.COLOR_JITTER_HUE,
            )
        if cfg is not None and cfg.GAMMA_JITTER:
            img = imutils.gamma_jitter(img, gamma_range=cfg.GAMMA_RANGE)
        if cfg is not None and cfg.GAUSSIAN_NOISE_STD > 0:
            img = imutils.gaussian_noise(img, max_std=cfg.GAUSSIAN_NOISE_STD_MAX if hasattr(cfg, 'GAUSSIAN_NOISE_STD_MAX') else cfg.GAUSSIAN_NOISE_STD)
        if cfg is not None and cfg.GAUSSIAN_BLUR_PROB > 0:
            img = imutils.gaussian_blur(img, prob=cfg.GAUSSIAN_BLUR_PROB)
        return img

    def _geometric_augment(self, img, label):
        cfg = self.cfg.AUG if self.cfg is not None else None
        if cfg is not None and cfg.RANDOM_RESIZED_CROP:
            img, label = imutils.random_resized_crop_seg(
                img,
                label,
                self.crop_size,
                scale=cfg.SCALE_RANGE,
                ratio=cfg.ASPECT_RANGE,
                ignore_index=self.ignore_index,
            )
        else:
            img, label = imutils.class_balanced_random_crop(
                img,
                label,
                crop_size=self.crop_size[0],
                ignore_index=self.ignore_index,
            )

        # Random flips/rotations
        if random.random() > 0.5:
            label = np.fliplr(label)
            img = np.fliplr(img)
        if random.random() > 0.5:
            label = np.flipud(label)
            img = np.flipud(img)

        k = random.randrange(4)
        if k:
            img = np.rot90(img, k).copy()
            label = np.rot90(label, k).copy()

        if cfg is not None and cfg.SMALL_ANGLE_ROT:
            img, label = imutils.small_angle_rotate(
                img,
                label,
                max_angle=cfg.MAX_ROTATION_DEG,
                crop_size=self.crop_size,
                ignore_index=self.ignore_index,
            )

        return img, label

    def _prepare(self, img, label, domain):
        # Normalize and to CHW tensors
        img = imutils.normalize_img(img)
        img = np.transpose(img, (2, 0, 1))

        img_tensor = torch.from_numpy(img.astype(np.float32))
        label_tensor = torch.from_numpy(label.astype(np.int64))
        domain_code = 1 if domain == 'urban' else 0 if domain == 'rural' else -1
        return img_tensor, label_tensor, domain_code

    def __getitem__(self, index):
        sample = self.samples[index]
        img = self.loader(sample['image']).astype(np.uint8)
        label = self.loader(sample['label']).astype(np.uint8)

        domain = sample['domain']

        if 'train' in self.mode:
            img, label = self._geometric_augment(img, label)
            img = self._photometric_augment(img, domain)
        else:
            # keep deterministic resize to crop size for eval
            img_pil = Image.fromarray(img.astype(np.uint8))
            label_pil = Image.fromarray(label.astype(np.uint8), mode='L')
            img = np.array(img_pil.resize((self.crop_size[1], self.crop_size[0]), Image.BILINEAR), dtype=np.float32)
            label = np.array(label_pil.resize((self.crop_size[1], self.crop_size[0]), Image.NEAREST), dtype=label.dtype)

        img_t, label_t, domain_code = self._prepare(img, label, domain)
        return {
            'image': img_t,
            'label': label_t,
            'domain': domain_code,
            'domain_name': domain,
            'id': sample['id'],
            'index': index,
        }

    def __len__(self):
        return len(self.samples)


class DomainBalancedSampler(Sampler):
    """Sampler that alternates urban/rural samples to get balanced batches."""

    def __init__(self, dataset, batch_size, drop_last=False, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.half_urban = batch_size // 2
        self.half_rural = batch_size - self.half_urban
        if self.half_urban == 0 and self.half_rural > 0:
            self.half_urban = 1
            self.half_rural = max(batch_size - 1, 0)

    def __iter__(self):
        urban = list(self.dataset.domain_to_indices.get('urban', []))
        rural = list(self.dataset.domain_to_indices.get('rural', []))
        if self.shuffle:
            random.shuffle(urban)
            random.shuffle(rural)

        batches = []
        while len(urban) >= self.half_urban and len(rural) >= self.half_rural:
            batch = [urban.pop() for _ in range(self.half_urban)] + [rural.pop() for _ in range(self.half_rural)]
            random.shuffle(batch)
            batches.extend(batch)

        if not self.drop_last:
            leftovers = urban + rural
            if self.shuffle:
                random.shuffle(leftovers)
            batches.extend(leftovers)

        return iter(batches)

    def __len__(self):
        urban = len(self.dataset.domain_to_indices.get('urban', []))
        rural = len(self.dataset.domain_to_indices.get('rural', []))
        if self.half_urban == 0 or self.half_rural == 0:
            length = urban + rural
            return length if not self.drop_last else (length // self.batch_size) * self.batch_size

        full_batches = min(urban // self.half_urban, rural // self.half_rural)
        length = full_batches * self.batch_size
        if not self.drop_last:
            length += (urban % self.half_urban) + (rural % self.half_rural)
        return length


def make_data_loader(args, cfg=None, **kwargs):
    if 'LOVEDA' in args.dataset:
        if 'train' in args.type:
            data_list = args.train_data_name_list
            dataset_path = args.train_dataset_path
            max_iters = args.max_iters
        else:
            data_list = getattr(args, 'test_data_name_list', [])
            dataset_path = args.test_dataset_path
            max_iters = None

        dataset = SemanticDatasetLOVEDA(
            dataset_path,
            data_list,
            args.crop_size,
            mode=args.type,
            cfg=cfg,
            max_iters=max_iters,
        )
        sampler = None
        use_balanced = cfg is not None and cfg.DATALOADER.DOMAIN_BALANCED_SAMPLER and 'train' in args.type
        if use_balanced:
            sampler = DomainBalancedSampler(dataset, batch_size=args.batch_size, drop_last=False, shuffle=args.shuffle)

        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=kwargs.get('num_workers', 16),
            drop_last=False,
        )
        return data_loader

    else:
        raise NotImplementedError
