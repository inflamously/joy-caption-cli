from collections import namedtuple
from typing import Any

import PIL.Image
import numpy as np
import tqdm
from PIL import ImageDraw
from PIL.Image import Image
from torch import Tensor

from segmentation.gen2seg_wrapper import get_device

# File: images_query.py
# Author: nflamously (using Kimi K2 llm)
# Original License: unknown

ClusterBox = namedtuple("ClusterBox", "color l u r d")
Point = namedtuple("Point", "x y")

import torch
from typing import List


def aspect_ratio_bboxes_from_points(points: List[Point], aspect_ratios=1 / 1, size=512) -> List[ClusterBox]:
    boxes = []
    width = int(round(size * aspect_ratios))
    height = size

    for pt in points:
        cx, cy = pt.x, pt.y
        half_w, half_h = width / 2.0, height / 2.0
        l = int(round(cx - half_w))
        u = int(round(cy - half_h))
        r = int(round(cx + half_w))
        d = int(round(cy + half_h))
        boxes.append(ClusterBox((0, 0, 0), l, u, r, d))
    return boxes


def autocrop_non_black(img: PIL.Image.Image, bg=(0, 0, 0)) -> PIL.Image.Image:
    mask = PIL.Image.new("1", img.size, 0)  # 1-bit mask
    mask.putdata([1 if px != bg else 0 for px in img.getdata()])
    bbox = mask.getbbox()
    return img.crop(bbox) if bbox else img


def multi_crop_images_from_bboxes(image: PIL.Image.Image, bboxes: List[ClusterBox]) -> List[PIL.Image.Image]:
    return [autocrop_non_black(image.crop((bbox.l, bbox.u, bbox.r, bbox.d))) for bbox in bboxes]


def nms_color_boxes(boxes: List[ClusterBox],
                    iou_threshold: float = 0.5) -> List[ClusterBox]:
    if not boxes:
        return []

    b = torch.tensor([[b.l, b.u, b.r, b.d] for b in boxes], dtype=torch.float32)
    areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    _, order = torch.sort(areas, descending=True)
    order = order.tolist()  # convert to plain Python ints
    keep = []

    while order:
        i = order.pop(0)
        keep.append(i)

        inter_l = torch.maximum(b[i, 0], b[order, 0])
        inter_u = torch.maximum(b[i, 1], b[order, 1])
        inter_r = torch.minimum(b[i, 2], b[order, 2])
        inter_d = torch.minimum(b[i, 3], b[order, 3])

        inter_w = (inter_r - inter_l).clamp(min=0)
        inter_h = (inter_d - inter_u).clamp(min=0)
        inter = inter_w * inter_h

        union = areas[i] + areas[order] - inter
        iou = inter / (union + 1e-6)

        order = [idx for j, idx in enumerate(order) if iou[j] <= iou_threshold]

    return [boxes[i] for i in keep]


def bboxes_to_points(bboxes: List[ClusterBox]):
    return [Point(round((bbox.l + bbox.r) / 2), round((bbox.u + bbox.d) / 2)) for bbox in bboxes]


def draw_bboxes(img: PIL.Image.Image, boxes: List[ClusterBox]) -> PIL.Image.Image:
    """
    Draw red rectangles for every ClusterBox on a copy of the image.
    """
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    for cb in boxes:
        draw.rectangle([cb.l, cb.u, cb.r - 1, cb.d - 1],
                       outline="red", width=3)
    return annotated


def draw_points(img: PIL.Image.Image, boxes: List[Point]) -> PIL.Image.Image:
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    for cb in boxes:
        draw.circle([cb.x, cb.y], radius=5, fill="red")
    return annotated


def unique_color_bboxes(img: PIL.Image.Image) -> List[ClusterBox]:
    """
    One tight bounding box per unique RGB value.
    """
    rgb = np.asarray(img.convert("RGB"))
    h, w, _ = rgb.shape

    # flatten to (N,3) and get unique colours + inverse map
    flat = rgb.reshape(-1, 3)
    colours, inv = np.unique(flat, axis=0, return_inverse=True)
    labels = inv.reshape(h, w)  # per-pixel cluster id

    boxes = []
    for cid, colour in enumerate(colours):
        mask = labels == cid
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        l, u, r, d = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        boxes.append(ClusterBox(tuple(colour.tolist()), l, u, r, d))
    return boxes


def kmeans_palette(colors: torch.Tensor,
                   t: float,
                   kmeans_iter: int = 30,
                   min_clusters: int = 2):
    """
    Split-colour k-means that never calls invalid indices and always
    returns `centers` and `labels` on the same device as `colors`.
    """
    device = colors.device
    colors = colors.detach().cpu()  # keep everything on CPU
    n_colors = colors.size(0)

    # -------------------------------------------------------------
    # 1.  k-means++ init (CPU-safe)
    # -------------------------------------------------------------
    centers = [colors[torch.randint(0, n_colors, (1,))].squeeze(0)]
    for _ in range(min(min_clusters - 1, n_colors - 1)):
        dists = torch.cdist(colors, torch.stack(centers))
        min_dists = dists.min(dim=1)[0]
        probs = (min_dists ** 2)
        probs /= probs.sum().clamp_min(1e-12)
        next_idx = torch.multinomial(probs, 1).item()
        centers.append(colors[next_idx])
    centers = torch.stack(centers)

    # -------------------------------------------------------------
    # 2.  k-means iterations
    # -------------------------------------------------------------
    labels = torch.empty(n_colors, dtype=torch.long, device="cpu")
    for _ in tqdm.trange(kmeans_iter):
        dists = torch.cdist(colors, centers)
        labels = dists.argmin(dim=1)

        new_centers = []
        for c in range(centers.size(0)):
            mask = labels == c
            if mask.any():
                new_centers.append(colors[mask].mean(dim=0))
            else:  # keep old center if empty
                new_centers.append(centers[c])
        new_centers = torch.stack(new_centers)

        if torch.allclose(centers, new_centers, atol=1e-4):
            break
        centers = new_centers

        # ---------------------------------------------------------
        # 3.  split worst cluster if radius > t
        # ---------------------------------------------------------
        worst_radius, worst_idx = 0.0, -1
        for c in range(centers.size(0)):
            mask = labels == c
            if mask.sum() > 1:
                r = torch.cdist(colors[mask], centers[c:c + 1]).max()
                if r > worst_radius:
                    worst_radius, worst_idx = r, c

        if worst_radius > t and centers.size(0) < n_colors:
            mask = labels == worst_idx
            pts = colors[mask]
            # two new seeds along the worst cluster's PCA direction
            mean_pt = pts.mean(dim=0)
            offset = (pts - mean_pt).std(dim=0) * 0.25
            c1 = mean_pt + offset
            c2 = mean_pt - offset
            centers = torch.cat([centers[:worst_idx], c1.unsqueeze(0),
                                 c2.unsqueeze(0), centers[worst_idx + 1:]])

    # move results back to the original device
    return centers.to(device), labels.to(device)


# WARNING: Following was completely LLM generated.
def kmeans_color_distance(image: PIL.Image.Image, threshold=50, kmeans_iter=10, min_clusters=2) -> tuple[
    Image, Tensor | Any, Tensor]:
    """
    Reduce the colour palette of a PIL Image by merging similar colours
    within `threshold` Euclidean distance (RGB space).

    Parameters
    ----------
    image : PIL.Image.Image
        Input image (RGB or RGBA).
    threshold : float
        Max L2 distance in RGB space below which colours are merged.

    Returns
    -------
    PIL.Image.Image
        New image with reduced palette.
    """
    device = get_device()

    # ------------------------------------------------------------------
    # 1. PIL → GPU tensor  (H,W,3)  float32
    # ------------------------------------------------------------------
    np_img = np.asarray(image.convert("RGB"), dtype=np.uint8)  # drop alpha if any
    img_t = torch.from_numpy(np_img).to(device, dtype=torch.float64)

    h, w, _ = img_t.shape
    flat = img_t.view(-1, 3)  # (N,3)

    # ------------------------------------------------------------------
    # 2. Unique palette + inverse map
    # ------------------------------------------------------------------
    palette, inv = torch.unique(flat, dim=0, return_inverse=True)  # (K,3), (N,)
    palette = palette.float()

    centers, labels = kmeans_palette(palette, threshold, kmeans_iter=kmeans_iter, min_clusters=min_clusters)

    # ------------------------------------------------------------------
    # 4. Map every pixel → quantised colour
    # ------------------------------------------------------------------
    new_flat = centers[labels][inv]  # (N,3)
    out_np = new_flat.view(h, w, 3).cpu().numpy()
    out_np = np.clip(out_np, 0, 255).astype(np.uint8)
    return PIL.Image.fromarray(out_np, mode="RGB"), centers, labels
