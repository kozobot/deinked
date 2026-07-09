"""Fine-tune a SegFormer to segment tattoos (the custom pixel-level localizer).

Trains ``nvidia/mit-b2`` (ImageNet-pretrained encoder, fresh 2-class head) on three data
sources with a curriculum, and saves an HF checkpoint to ``data/models/tattoo-segformer/`` for
``deink.TattooMaskSegmenter`` to load. Everything is bare ``torch`` + ``torchvision`` — no
lightning/fastai, no new deps, matching the repo's transformers-native constraint.

Data sources (built by the sibling scripts):
  - curated synthetic  : data/silhouette/{source,mask}/synthetic-*.png      (mask 0..255)
  - generated synthetic: data/silhouette/synthetic_gen/{img,mask}/gen-*.png (mask 0/255)
  - derived real       : data/deinked/rawdata/<stem>_tattoo.* + data/silhouette/derived/<stem>.png
                         (trimap 0/1/255; only stems in derived/split.json['train'])

Curriculum: Stage A warms up on the unlimited-but-synthetic sources; Stage B fine-tunes on a
weighted mix that leans on the real (but noisier) diff-derived masks. The encoder is frozen for
the first ``--freeze-epochs`` so the fresh head settles before the backbone moves.

Loss = class-weighted CrossEntropy(ignore_index=255) + Dice, both respecting the ignore band.
bf16 autocast (native on Blackwell sm_120 — no GradScaler).

Usage:
    python scripts/train_tattooseg.py                 # full run
    python scripts/train_tattooseg.py --smoke         # 2 epochs, tiny subset (verification)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys

# Reduce allocator fragmentation on a shared/constrained GPU — must be set before CUDA init.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2
from torchvision import tv_tensors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deink.utils import get_device

IGNORE = 255
OUT_DIR = "data/models/tattoo-segformer"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- data
def build_items(rawdata="data/deinked/rawdata", derived="data/silhouette/derived"):
    """Return {'synthetic': [...], 'real': [...]} of (image_path, mask_path, kind) tuples.

    kind: 'binary' (mask 0..255, threshold 127) or 'trimap' (already 0/1/255).
    """
    synthetic, real = [], []

    # curated synthetic (source/mask paired by filename)
    for src in sorted(glob.glob("data/silhouette/source/*.png")):
        m = os.path.join("data/silhouette/mask", os.path.basename(src))
        if os.path.isfile(m):
            synthetic.append((src, m, "binary"))
    # generated synthetic
    for img in sorted(glob.glob("data/silhouette/synthetic_gen/img/*.png")):
        m = os.path.join("data/silhouette/synthetic_gen/mask", os.path.basename(img))
        if os.path.isfile(m):
            synthetic.append((img, m, "binary"))
    # derived real (train split only)
    split_path = os.path.join(derived, "split.json")
    train_stems = set(json.load(open(split_path))["train"]) if os.path.isfile(split_path) else set()
    for m in sorted(glob.glob(os.path.join(derived, "*.png"))):
        stem = os.path.splitext(os.path.basename(m))[0]
        if stem not in train_stems:
            continue
        hits = glob.glob(os.path.join(rawdata, f"{stem}_tattoo.*"))
        if hits:
            real.append((hits[0], m, "trimap"))

    return {"synthetic": synthetic, "real": real}


def _load_label(path: str, kind: str) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    if kind == "binary":
        return (arr > 127).astype(np.uint8)
    return arr  # trimap: already 0/1/255


class TattooSegDataset(Dataset):
    """(image, label) pairs at ``size`` px with foreground-biased cropping + augmentation."""

    def __init__(self, items, size=512, train=True):
        self.items = items
        self.size = size
        self.train = train
        self.geo = v2.Compose([
            v2.RandomHorizontalFlip(0.5),
            v2.RandomRotation(15),
            v2.RandomPerspective(distortion_scale=0.2, p=0.3),
        ])
        # Photometric augments (image only) are what bridge synthetic -> real.
        self.photo = v2.Compose([
            v2.ColorJitter(0.3, 0.3, 0.3, 0.05),
            v2.RandomGrayscale(0.1),
            v2.RandomApply([v2.GaussianBlur(5)], p=0.2),
        ])
        self.norm = v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self):
        return len(self.items)

    def _crop(self, img: Image.Image, label: np.ndarray):
        """Train: foreground-biased square crop → ``size``. Eval: resize the *whole* image to
        ``size`` (no crop) so it matches inference (SegformerImageProcessor resizes the full
        frame) and no tattoo is ever cropped out of the val score — the latter was a real source
        of val noise, since a center-square crop discards the edges of portrait photos."""
        if not self.train:
            img = img.resize((self.size, self.size), Image.BILINEAR)
            lab = Image.fromarray(label).resize((self.size, self.size), Image.NEAREST)
            return img, np.array(lab)
        W, H = img.size
        fg = np.argwhere(label == 1)
        if len(fg) and random.random() < 0.7:
            cy, cx = fg[random.randrange(len(fg))]
            s = int(min(H, W) * random.uniform(0.4, 1.0))
        else:
            cy, cx, s = H // 2, W // 2, min(H, W)
        x0 = int(np.clip(cx - s // 2, 0, max(0, W - s)))
        y0 = int(np.clip(cy - s // 2, 0, max(0, H - s)))
        img = img.crop((x0, y0, x0 + s, y0 + s)).resize((self.size, self.size), Image.BILINEAR)
        lab = Image.fromarray(label[y0:y0 + s, x0:x0 + s]).resize(
            (self.size, self.size), Image.NEAREST)
        return img, np.array(lab)

    def __getitem__(self, i):
        img_path, mask_path, kind = self.items[i]
        img = Image.open(img_path).convert("RGB")
        label = _load_label(mask_path, kind)
        if label.shape != img.size[::-1]:  # guard against any size drift
            label = np.array(Image.fromarray(label).resize(img.size, Image.NEAREST))
        img, label = self._crop(img, label)

        image = tv_tensors.Image(torch.from_numpy(np.array(img)).permute(2, 0, 1))
        mask = tv_tensors.Mask(torch.from_numpy(label.astype(np.int64))[None])
        if self.train:
            image, mask = self.geo(image, mask)
            image = self.photo(image)
        image = self.norm(image.float() / 255.0)
        return image, mask[0].long()


# --------------------------------------------------------------------------- loss
def dice_loss(logits, target, ignore=IGNORE, eps=1.0):
    """Soft Dice on the foreground channel, excluding ignore pixels."""
    prob = logits.softmax(dim=1)[:, 1]
    valid = (target != ignore).float()
    tgt = (target == 1).float() * valid
    prob = prob * valid
    inter = (prob * tgt).sum(dim=(1, 2))
    denom = prob.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2))
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def seg_loss(logits, target, fg_weight):
    w = torch.tensor([1.0, fg_weight], device=logits.device)
    ce = F.cross_entropy(logits, target, weight=w, ignore_index=IGNORE)
    return ce + dice_loss(logits, target)


@torch.no_grad()
def evaluate(model, loader, device):
    """Foreground IoU + Dice on a held-out loader (ignore pixels excluded)."""
    model.eval()
    inter = union = tp = fp_fn = 0
    for image, target in loader:
        image, target = image.to(device), target.to(device)
        logits = _upsample(model(pixel_values=image).logits, target.shape[-2:])
        pred = logits.argmax(1)
        valid = target != IGNORE
        p, t = (pred == 1) & valid, (target == 1) & valid
        inter += (p & t).sum().item()
        union += (p | t).sum().item()
        tp += (p & t).sum().item()
        fp_fn += (p ^ t).sum().item()
    iou = inter / (union + 1e-9)
    dice = 2 * tp / (2 * tp + fp_fn + 1e-9)
    return iou, dice


def _upsample(logits, size):
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


# --------------------------------------------------------------------------- train
def make_loader(items, size, batch, train, weights=None):
    ds = TattooSegDataset(items, size=size, train=train)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True) if weights else None
    return DataLoader(ds, batch_size=batch, shuffle=(train and sampler is None),
                      sampler=sampler, num_workers=4, drop_last=train, pin_memory=True)


def set_encoder_requires_grad(model, flag):
    # ``model.segformer`` IS the hierarchical encoder backbone; ``model.decode_head`` is the head.
    for p in model.segformer.parameters():
        p.requires_grad = flag


def run_stage(model, loader, val_loader, device, epochs, lr_enc, lr_head, fg_weight,
              freeze_epochs, tag, best_dice, out_dir, processor, accum=1, patience=None):
    enc_params = list(model.segformer.parameters())
    head_params = list(model.decode_head.parameters())
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": lr_enc}, {"params": head_params, "lr": lr_head}],
        weight_decay=0.01)
    # One scheduler step per *optimizer* step (i.e. per accumulation window), not per batch.
    opt_steps_per_epoch = math.ceil(len(loader) / accum)
    steps = max(1, epochs * opt_steps_per_epoch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[lr_enc, lr_head], total_steps=steps, pct_start=0.1)

    since_improve = 0
    for ep in range(epochs):
        set_encoder_requires_grad(model, ep >= freeze_epochs)
        model.train()
        running = 0.0
        n_batches = len(loader)
        opt.zero_grad()
        for i, (image, target) in enumerate(loader):
            image, target = image.to(device), target.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                logits = _upsample(model(pixel_values=image).logits, target.shape[-2:])
                loss = seg_loss(logits.float(), target, fg_weight)
            (loss / accum).backward()  # scale so accumulated grads match a true large batch
            if (i + 1) % accum == 0 or (i + 1) == n_batches:
                opt.step()
                sched.step()
                opt.zero_grad()
            running += loss.item()
        line = f"[{tag}] epoch {ep + 1}/{epochs} loss={running / n_batches:.4f}"
        if val_loader is not None:
            iou, dice = evaluate(model, val_loader, device)
            line += f"  val_IoU={iou:.3f} val_Dice={dice:.3f}"
            if dice > best_dice:
                best_dice = dice
                _save(model, processor, out_dir)
                line += "  *saved"
                since_improve = 0
            else:
                since_improve += 1
        print(line, flush=True)
        # Early stop once val Dice has plateaued — training loss keeps falling well past the
        # point where the model overfits (val Dice collapsed ~epoch 8→25 in an earlier run), so
        # patience both saves time and avoids selecting a lucky late epoch.
        if patience and since_improve >= patience:
            print(f"[{tag}] early stop: no val improvement in {patience} epochs", flush=True)
            break
    return best_dice


def _save(model, processor, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="nvidia/mit-b2", help="use nvidia/mit-b1 if it overfits")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4, help="per-step batch; lower on a busy GPU")
    ap.add_argument("--grad-accum", type=int, default=2,
                    help="accumulate this many batches per optimizer step (effective batch = batch*grad-accum)")
    ap.add_argument("--no-grad-checkpoint", action="store_true",
                    help="disable gradient checkpointing (faster, but much more VRAM)")
    ap.add_argument("--epochs-pretrain", type=int, default=15)
    ap.add_argument("--epochs-finetune", type=int, default=25)
    ap.add_argument("--freeze-epochs", type=int, default=2)
    ap.add_argument("--lr-enc", type=float, default=6e-5)
    ap.add_argument("--lr-head", type=float, default=6e-4)
    ap.add_argument("--fg-weight", type=float, default=3.0)
    ap.add_argument("--real-frac", type=float, default=0.6, help="target real fraction in Stage B")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="fraction of real data held out for validation / model selection")
    ap.add_argument("--patience", type=int, default=6,
                    help="stop the fine-tune stage after this many epochs without val improvement")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="2 epochs on a tiny subset")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()

    items = build_items()
    print(f"data: {len(items['synthetic'])} synthetic, {len(items['real'])} real")
    if not items["synthetic"] and not items["real"]:
        raise SystemExit("No training data. Run derive_masks.py and gen_synthetic_seg.py first.")

    if args.smoke:
        items = {"synthetic": items["synthetic"][:12], "real": items["real"][:12]}
        args.epochs_pretrain = args.epochs_finetune = 1
        args.freeze_epochs = 0
        args.batch = min(args.batch, 4)

    # Held-out val for model selection: a fixed fraction of real (falls back to synthetic if no
    # real data). A larger, stable val set makes the "best" checkpoint less of a lucky epoch —
    # the earlier 10% (~6 images) split was noisy enough to swing val Dice 0.88 -> 0.07.
    pool = items["real"] or items["synthetic"]
    random.shuffle(pool)
    n_val = max(2, int(len(pool) * args.val_frac))
    val_items, real_train = pool[:n_val], (items["real"][n_val:] if items["real"] else [])
    val_loader = make_loader(val_items, args.size, args.batch, train=False)
    print(f"val: {len(val_items)} held-out real images")

    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    id2label = {0: "background", 1: "tattoo"}
    processor = SegformerImageProcessor(do_reduce_labels=False)  # keep the background class!
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.encoder, num_labels=2, id2label=id2label,
        label2id={v: k for k, v in id2label.items()}, ignore_mismatched_sizes=True,
    ).to(device)
    if not args.no_grad_checkpoint:
        # Trades ~20% compute for a large activation-memory saving — the main OOM lever on a
        # 16 GB (or shared) card. use_reentrant=False so it works even while the encoder is frozen.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    best = -1.0
    # Stage A — synthetic warmup.
    if items["synthetic"]:
        loader = make_loader(items["synthetic"], args.size, args.batch, train=True)
        best = run_stage(model, loader, val_loader, device, args.epochs_pretrain,
                         args.lr_enc, args.lr_head, args.fg_weight, args.freeze_epochs,
                         "pretrain", best, args.out, processor, accum=args.grad_accum)

    # Stage B — real + synthetic mix via a weighted sampler that hits real_frac on average.
    combined = real_train + items["synthetic"]
    if combined:
        n_real, n_syn = len(real_train), len(items["synthetic"])
        wr = (args.real_frac / n_real) if n_real else 0.0
        ws = ((1 - args.real_frac) / n_syn) if n_syn else 0.0
        weights = [wr] * n_real + [ws] * n_syn
        loader = make_loader(combined, args.size, args.batch, train=True, weights=weights)
        best = run_stage(model, loader, val_loader, device, args.epochs_finetune,
                         args.lr_enc, args.lr_head, args.fg_weight, args.freeze_epochs,
                         "finetune", best, args.out, processor, accum=args.grad_accum,
                         patience=args.patience)

    # Always persist a final checkpoint (in case val never improved on a smoke run).
    if best < 0:
        _save(model, processor, args.out)
    with open(os.path.join(args.out, "training_meta.json"), "w") as f:
        json.dump({
            "encoder": args.encoder, "size": args.size, "seed": args.seed,
            "batch": args.batch, "grad_accum": args.grad_accum,
            "grad_checkpoint": not args.no_grad_checkpoint,
            "epochs_pretrain": args.epochs_pretrain, "epochs_finetune": args.epochs_finetune,
            "fg_weight": args.fg_weight, "real_frac": args.real_frac,
            "val_frac": args.val_frac, "patience": args.patience, "n_val": len(val_items),
            "n_synthetic": len(items["synthetic"]), "n_real": len(items["real"]),
            "best_val_dice": best, "smoke": args.smoke,
        }, f, indent=2)
    print(f"saved checkpoint + training_meta.json -> {args.out}  (best val Dice {best:.3f})")


if __name__ == "__main__":
    main()
