import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


def tta_logits(model, image_batch: torch.Tensor, use_tta: bool = True) -> torch.Tensor:
    """
    Run the model with simple flip-based TTA and return averaged logits.

    Args:
        model: segmentation model.
        image_batch: (B, C, H, W) tensor.
        use_tta: if False, just run model once and return logits.

    Returns:
        logits: (B, num_classes, H, W) averaged over TTA variants.
    """
    if not use_tta:
        with torch.no_grad():
            return model(image_batch)

    logits_list = []
    with torch.no_grad():
        # original
        logits = model(image_batch)
        logits_list.append(logits)

        # horizontal flip
        img_h = torch.flip(image_batch, dims=[3])
        logits_h = model(img_h)
        logits_h = torch.flip(logits_h, dims=[3])
        logits_list.append(logits_h)

        # vertical flip
        img_v = torch.flip(image_batch, dims=[2])
        logits_v = model(img_v)
        logits_v = torch.flip(logits_v, dims=[2])
        logits_list.append(logits_v)

    stacked = torch.stack(logits_list, dim=0)
    return stacked.mean(dim=0)


def _connected_components(binary_mask: torch.Tensor, connectivity: int = 4):
    """
    Simple CPU connected components for a single-channel binary mask.
    Returns (label_map, num_components).
    """
    if binary_mask.numel() == 0:
        return binary_mask, 0
    np_mask = binary_mask.cpu().numpy().astype("uint8")
    try:
        import scipy.ndimage as ndi

        structure = None
        if connectivity == 8:
            structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        labeled, num = ndi.label(np_mask, structure=structure)
        return torch.from_numpy(labeled), int(num)
    except Exception:
        # fallback flood-fill
        h, w = np_mask.shape
        labels = torch.zeros((h, w), dtype=torch.int32)
        current = 0
        for y in range(h):
            for x in range(w):
                if np_mask[y, x] == 0 or labels[y, x] != 0:
                    continue
                current += 1
                stack = [(y, x)]
                labels[y, x] = current
                while stack:
                    cy, cx = stack.pop()
                    for ny, nx in [(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)]:
                        if 0 <= ny < h and 0 <= nx < w and np_mask[ny, nx] == 1 and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
        return labels, current


def _majority_neighbor_class(pred_mask: torch.Tensor, coords, num_classes: int, ignore_class: int):
    """
    coords: list of (y, x) tuples to reassign.
    """
    if len(coords) == 0:
        return
    h, w = pred_mask.shape
    np_mask = pred_mask.cpu().numpy()
    for (y, x) in coords:
        y0, y1 = max(y - 1, 0), min(y + 2, h)
        x0, x1 = max(x - 1, 0), min(x + 2, w)
        patch = np_mask[y0:y1, x0:x1].ravel()
        counts = torch.bincount(torch.tensor(patch, dtype=torch.int64), minlength=num_classes)
        if ignore_class < counts.numel():
            counts[ignore_class] = 0
        new_class = int(torch.argmax(counts).item()) if counts.sum() > 0 else 0
        np_mask[y, x] = new_class
    pred_mask.copy_(torch.from_numpy(np_mask))


def _binary_morphology(mask: torch.Tensor, op: str, kernel_size: int = 3):
    """mask: (H, W) float/byte tensor, op in {'dilate','erode'}."""
    pad = kernel_size // 2
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device, dtype=torch.float32)
    x = mask.float().unsqueeze(0).unsqueeze(0)
    conv = F.conv2d(x, kernel, padding=pad)
    if op == "dilate":
        return (conv > 0).float().squeeze()
    elif op == "erode":
        return (conv >= kernel.numel()).float().squeeze()
    else:
        raise ValueError("Unsupported op {}".format(op))


def clean_mask(pred_mask: torch.Tensor, min_areas, apply_open_close: bool = True) -> torch.Tensor:
    """
    Post-process a single predicted mask with small-component removal and
    optional morphological opening/closing.

    Args:
        pred_mask: (H, W) tensor of int64 class ids.
        min_areas: dict or list mapping class_id -> min area in pixels.
        apply_open_close: whether to apply simple opening/closing for solid classes.

    Returns:
        refined_mask: (H, W) tensor of int64 class ids.
    """
    device = pred_mask.device
    refined = pred_mask.detach().clone().cpu()
    min_area_map = {}
    if isinstance(min_areas, dict):
        min_area_map = {int(k): int(v) for k, v in min_areas.items()}
    elif isinstance(min_areas, (list, tuple)):
        min_area_map = {i: int(v) for i, v in enumerate(min_areas)}

    num_classes = int(refined.max().item() + 1)
    if min_area_map:
        num_classes = max(num_classes, max(min_area_map.keys()) + 1)

    for cls in range(num_classes):
        cls_min_area = min_area_map.get(cls, 0)
        if cls_min_area <= 0:
            continue
        binary = (refined == cls)
        if not binary.any():
            continue
        labels, comp_num = _connected_components(binary, connectivity=4)
        if comp_num == 0:
            continue
        labels = labels.to(torch.int64)
        for comp_id in range(1, comp_num + 1):
            coords = torch.nonzero(labels == comp_id, as_tuple=False)
            area = coords.shape[0]
            if area < cls_min_area:
                coords_list = [(int(y), int(x)) for y, x in coords]
                _majority_neighbor_class(refined, coords_list, num_classes=num_classes, ignore_class=cls)

    if apply_open_close:
        solid_classes = [c for c in range(num_classes) if c != 0]
        updated = refined.clone()
        for cls in solid_classes:
            binary = (refined == cls).float()
            if binary.sum() == 0:
                continue
            closed = _binary_morphology(_binary_morphology(binary, "dilate"), "erode")
            opened = _binary_morphology(_binary_morphology(closed, "erode"), "dilate")
            mask_cls = (opened > 0.5)
            updated[mask_cls] = cls
        refined = updated

    return refined.to(device=device, dtype=pred_mask.dtype)


def crf_refine(image_np, prob_np, n_iters: int = 5, sxy: int = 3, srgb: int = 5):
    """
    Optional DenseCRF refinement. Returns refined (H, W) int mask.
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except Exception:
        # If CRF is unavailable, fall back to argmax of prob_np
        return prob_np.argmax(axis=0).astype("int64")

    num_classes, H, W = prob_np.shape
    d = dcrf.DenseCRF2D(W, H, num_classes)
    U = unary_from_softmax(prob_np)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=sxy, compat=3)
    d.addPairwiseBilateral(sxy=sxy, srgb=srgb, rgbim=image_np, compat=5)
    Q = d.inference(n_iters)
    refined = torch.tensor(Q, dtype=torch.float32).view(num_classes, H, W).argmax(dim=0).numpy().astype("int64")
    return refined
