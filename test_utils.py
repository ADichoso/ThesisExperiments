import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image, ImageEnhance
import os
import matplotlib.patches as mpatches

hide_title = True

img_root = "Datasets/ACOD-12K/Test/Imgs/"
gt_root = "Datasets/ACOD-12K/Test/GT/"
pred_root = "Results/E100/"
figures_root = "Figures/"
data_root = "Data/"


def get_img_root():
    return img_root

def get_gt_root():
    return gt_root

def display_images(images=3, image_files=None, title=None, save_path=None,
                   networks=None, pred_network_roots=None,
                   wspace=0.05, hspace=0.05,
                   title_fontsize=32, label_fontsize=20):
    global img_root, gt_root

    random_imgs = random.sample(image_files, images)
    cols = 2 + len(networks)

    # Get aspect ratio from a sample image
    sample_img = Image.open(os.path.join(img_root, random_imgs[0]))
    img_w, img_h = sample_img.size
    aspect = img_h / img_w

    # Include spacing in grid layout
    total_width = cols + (cols - 1) * wspace
    total_height = images * aspect + (images - 1) * hspace

    fig = plt.figure(figsize=(total_width * 2.5, total_height * 2.5))

    for i in range(images):
        file_name = random_imgs[i]
        img = Image.open(os.path.join(img_root, file_name))
        gt = Image.open(os.path.join(gt_root, file_name.replace(".jpg", ".png")))
        preds = [Image.open(os.path.join(pred_root, file_name.replace(".jpg", ".png"))) for pred_root in pred_network_roots]
        imgs = [img, gt] + preds

        for j, im in enumerate(imgs):
            cell_width = 1.0 / total_width
            cell_height = aspect / total_height

            left = j * (1 + wspace) * cell_width
            bottom = 1.0 - (i + 1) * cell_height - i * hspace * cell_height

            ax = fig.add_axes([left, bottom, cell_width, cell_height])
            ax.imshow(im)
            ax.axis("off")

            if i == 0:
                ax.set_title(
                    "Image" if j == 0 else
                    "Ground Truth" if j == 1 else
                    networks[j - 2],
                    fontsize=label_fontsize
                )

    if title and not hide_title:
        fig.suptitle(title, fontsize=title_fontsize, y=1.1)

    if save_path:
        fig.savefig(os.path.join(figures_root, save_path), dpi=100, bbox_inches='tight')

    plt.show()

def overlay_masks(img, gt, pred=None, alpha=0.8, darken_bg=0.6):
    """
    Overlay masks on an image with high-contrast colors.
    - Cyan: TP (GT=1 & Pred=1)
    - Orange: FN (GT=1 & Pred=0)
    - Magenta: FP (GT=0 & Pred=1)
    If pred=None, only overlay GT (orange).
    Background darkens only in unmasked regions.
    """
    gt = np.array(gt.convert("L")) > 127
    pred = np.array(pred.convert("L")) > 127 if pred is not None else None

    # Convert to array
    img = np.array(img.convert("RGB")).astype(np.float32)

    # Define mask regions
    if pred is not None:
        tp = gt & pred
        fn = gt & ~pred
        fp = ~gt & pred
        covered = tp | fn | fp
    else:
        tp = fn = fp = None
        covered = gt

    # Darken background only outside mask
    img_darken = img * darken_bg
    img[~covered] = img_darken[~covered]

    # Overlay RGBA mask
    overlay = np.zeros((gt.shape[0], gt.shape[1], 4), dtype=np.uint8)
    if pred is not None:
        overlay[tp] = [0, 255, 255, int(225*alpha)]      # TP: cyan
        overlay[fn] = [255, 165, 0, int(225*alpha)]      # FN: orange
        overlay[fp] = [255, 0, 255, int(225*alpha)]      # FP: magenta
    else:
        overlay[gt] = [65, 105, 225, int(225*alpha)]      # GT only: blue

    return img.astype(np.uint8), overlay

def display_images2(images=3, image_files=None, title=None, save_path=None,
                    networks=None, pred_network_roots=None,
                    wspace=0.05, hspace=0.05,
                    title_fontsize=32, label_fontsize=20):
    global img_root, gt_root

    random_imgs = random.sample(image_files, images)
    cols = 1 + len(networks) + 1  # label + GT overlay + prediction overlays

    # Aspect ratio based on first image
    sample_img = Image.open(os.path.join(img_root, random_imgs[0]))
    img_w, img_h = sample_img.size
    aspect = img_h / img_w
    total_width = cols + (cols - 1) * wspace
    total_height = images * aspect + (images - 1) * hspace
    fig = plt.figure(figsize=(total_width * 2.5, total_height * 2.5))

    for i, file_name in enumerate(random_imgs):
        img = Image.open(os.path.join(img_root, file_name))
        gt = Image.open(os.path.join(gt_root, file_name.replace(".jpg", ".png")))
        preds = [Image.open(os.path.join(pred_root, file_name.replace(".jpg", ".png"))) 
                 for pred_root in pred_network_roots]

        # Extract crop name from filename
        crop_name = os.path.splitext(file_name)[0]

        cell_width = 1.0 / total_width
        cell_height = aspect / total_height
        bottom = 1.0 - (i + 1) * cell_height - i * hspace * cell_height

        # Label column (just text)
        left = 0
        ax = fig.add_axes([left, bottom, cell_width / 0.1, cell_height])
        ax.text(0.09, 0.5, crop_name.split("_")[0].upper(), fontsize=label_fontsize / 1.2,
                ha='center', va='center', rotation=90)
        ax.axis("off")
            
        # GT + predictions
        for j, pred in enumerate([None] + preds):
            left = (1 + j * (1 + wspace)) * cell_width
            ax = fig.add_axes([left, bottom, cell_width, cell_height])
            base, overlay = overlay_masks(img, gt, pred)
            ax.imshow(base)
            ax.imshow(overlay)
            ax.axis("off")
            if i == 0:
                if pred is None:
                    ax.set_title("Ground Truth", fontsize=label_fontsize)
                else:
                    ax.set_title(networks[j-1], fontsize=label_fontsize)

    if title:
        fig.suptitle(title, fontsize=title_fontsize, y=1.02)

    # Legend
    gt_patch = mpatches.Patch(color=(65/255, 105/255, 225/255), label='GT (Ground Truth)')
    tp_patch = mpatches.Patch(color=(0, 1, 1), label='TP (True Positive)')
    fn_patch = mpatches.Patch(color=(1, 165/255, 0), label='FN (False Negative)')
    fp_patch = mpatches.Patch(color=(1, 0, 1), label='FP (False Positive)')

    fig.legend(
        handles=[gt_patch, tp_patch, fn_patch, fp_patch],
        loc='center',
        bbox_to_anchor=(0.55, -0.06),
        ncol=4,
        fontsize=label_fontsize - 2,
        frameon=True,
        title="Mask Legend",
        title_fontsize=label_fontsize - 2
    )

    plt.subplots_adjust(bottom=0.04)
    if save_path:
        fig.savefig(os.path.join(figures_root, save_path), dpi=100, bbox_inches='tight')
    plt.show()

def display_gt_two_columns(
    row1_types,
    row2_types,
    images_per_type=3,
    image_files=None,
    title=None,
    save_path=None,
    wspace=0.01,
    hspace=0.13,
    title_fontsize=32,
    label_fontsize=10
):
    """
    Display GT overlays in TWO COLUMNS.

    Column 1 -> crops in row1_types
    Column 2 -> crops in row2_types

    image_files: flat filename list
    """

    global img_root, gt_root

    # -------------------------------------------------
    # 🔹 GROUP FILES BY CROP PREFIX
    # -------------------------------------------------
    files_by_crop = {}
    for fname in image_files:
        crop = fname.split("_")[0].lower()
        files_by_crop.setdefault(crop, []).append(fname)

    # -------------------------------------------------
    # 🔹 COLLECT FILES PER COLUMN
    # -------------------------------------------------
    def collect_files(crop_types):
        collected = []
        for crop in crop_types:
            key = crop.lower()
            if key not in files_by_crop:
                print(f"[WARN] No files for '{crop}'")
                continue

            files = files_by_crop[key]
            n_sample = min(images_per_type, len(files))
            collected.extend(random.sample(files, n_sample))
        return collected

    col1_files = collect_files(row1_types)
    col2_files = collect_files(row2_types)

    max_rows = max(len(col1_files), len(col2_files))
    cols = 2

    # --- aspect ratio ---
    sample_file = image_files[0]
    sample_img = Image.open(os.path.join(img_root, sample_file))
    img_w, img_h = sample_img.size
    aspect = img_h / img_w

    total_width = cols + (cols - 1) * wspace
    total_height = max_rows * aspect + (max_rows - 1) * hspace

    fig = plt.figure(figsize=(cols * 2.5, total_height * 2.5))

    # =================================================
    # 🔹 DRAW GRID
    # =================================================
    columns_data = [col1_files, col2_files]

    for col_idx, col_files in enumerate(columns_data):
        for row_idx in range(len(col_files)):

            file_name = col_files[row_idx]
            crop_name = file_name.split("_")[0].upper()

            img = Image.open(os.path.join(img_root, file_name))
            gt = Image.open(os.path.join(gt_root, file_name))

            cell_width = 1.0 / total_width
            cell_height = aspect / total_height

            left = col_idx * (1 + wspace) * cell_width
            bottom = (
                1.0
                - (row_idx + 1) * cell_height
                - row_idx * hspace * cell_height
            )

            ax = fig.add_axes([left, bottom, cell_width, cell_height])

            base, overlay = overlay_masks(img, gt, pred=None)

            ax.imshow(base)
            ax.imshow(overlay)
            ax.axis("off")

            # ✅ Make room for caption
            ax.set_ylim(ax.get_ylim()[0] + 10, ax.get_ylim()[1])

            # ✅ Label under EVERY image (not clipped)
            label_y = bottom - (0.035 * cell_height)

            fig.text(
                left + cell_width / 2,
                label_y,
                crop_name,
                ha="center",
                va="top",
                fontsize=label_fontsize - 2,
                weight="bold"
            )

    if title:
        fig.suptitle(title, fontsize=title_fontsize, y=1.06)

    if save_path:
        fig.savefig(os.path.join(figures_root, save_path),
                    dpi=100, bbox_inches='tight')

    plt.show()

def display_images3(
    images=2,                  # samples per crop
    image_files=None,          # dict: crop -> list[str]
    crops=None,
    title=None,
    save_path=None,
    networks=None,
    pred_network_roots=None,   # [net][crop]
    wspace=0.05,
    hspace=0.05,
    title_fontsize=32,
    label_fontsize=36
):

    global img_root, gt_root

    num_crops = len(crops)
    num_networks = 1 + len(networks)   # GT + networks
    rows = num_crops * images
    cols = num_networks + 1  # +1 for dedicated label column

    # sample images per crop
    crop_imgs = {
        crop: random.sample(image_files[crop], images)
        for crop in crops
    }

    # aspect ratio from a sample image
    sample_img = Image.open(os.path.join(img_root, crop_imgs[crops[0]][0]))
    img_w, img_h = sample_img.size
    aspect = img_h / img_w

    figsize = (cols * 3, rows * 3 * aspect)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_2d(axes)

    # hide dedicated label column axes
    for row in range(rows):
        axes[row, 0].axis("off")

    for crop_idx, crop in enumerate(crops):
        for img_idx, file_name in enumerate(crop_imgs[crop]):
            row = crop_idx * images + img_idx

            img = Image.open(os.path.join(img_root, file_name))
            gt = Image.open(os.path.join(gt_root, file_name.replace(".jpg", ".png")))

            # ---------- GT COLUMN ----------
            ax = axes[row, 1]  # start from column 1
            base, overlay = overlay_masks(img, gt)
            ax.imshow(base)
            ax.imshow(overlay)
            ax.axis("off")
            if row == 0:
                ax.set_title("Ground Truth", fontsize=label_fontsize)

            # ---------- NETWORK COLUMNS ----------
            for net_idx, net_name in enumerate(networks):
                pred = Image.open(
                    os.path.join(
                        pred_network_roots[net_idx],
                        file_name.replace(".jpg", ".png")
                    )
                )
                ax = axes[row, net_idx + 2]  # shift by 2: 0=label, 1=GT
                base, overlay = overlay_masks(img, gt, pred)
                ax.imshow(base)
                ax.imshow(overlay)
                ax.axis("off")
                if row == 0:
                    ax.set_title(net_name, fontsize=label_fontsize)

        # ---------- CROP LABEL IN DEDICATED COLUMN ----------
        mid_row = crop_idx * images + images // 2
        axes[mid_row, 0].text(
            .9, 1, crop.upper(),
            fontsize=label_fontsize,
            rotation=90,
            ha="center",
            va="center",
            transform=axes[mid_row, 0].transAxes
        )

        # Create legend patches
    gt_patch = mpatches.Patch(color=(65/255, 105/255, 225/255), label='GT (Ground Truth)') # Blue (GT)
    tp_patch = mpatches.Patch(color=(0, 1, 1), label='TP (True Positive)') #Cyan
    fn_patch = mpatches.Patch(color=(1, 165/255, 0), label='FN (False Negative)')     # Orange
    fp_patch = mpatches.Patch(color=(1, 0, 1), label='FP (False Positive)')           # Magenta
    # Add horizontal legend below all subplots
    fig.legend(
        handles=[gt_patch, tp_patch, fn_patch, fp_patch],
        loc='center',
        bbox_to_anchor=(0.55, 0),   # slightly below the figure
        ncol=4,                        # one row, three columns
        fontsize=label_fontsize - 2,
        frameon=True,
        title="Mask Legend",
        title_fontsize=label_fontsize - 2
    )
    
    # ---------- FIGURE LAYOUT ----------
    plt.subplots_adjust(
        left=0.1,
        right=0.95,
        top=0.95,
        bottom=0.05,
        wspace=wspace,
        hspace=hspace
    )

    if title and not hide_title:
        fig.suptitle(title, fontsize=title_fontsize)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

