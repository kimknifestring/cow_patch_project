"""학습에 안 쓴 이미지로 패치 평가 + 다른 모델로 전이율 측정.

조건 3가지 (같은 무작위 변형, 시드 고정):
  No patch / Random noise patch (같은 크기) / Our patch
성공률 = top-1 예측이 banana인 비율

    python evaluate.py --patch outputs/banana_patch.pt
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from patch_attack.common import (AREA_RANGE, DEVICE, IMG_SIZE, LABELS, MODELS, ROT_DEG, TARGET_CLASS,
                                 apply_patch, eval_dataset, load_model, sample_transforms)

CURVE_AREAS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="outputs/banana_patch.pt")
    ap.add_argument("--images", type=int, default=1000, help="평가에 쓸 val 이미지 수")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--rot", type=float, default=ROT_DEG)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="outputs")
    return ap.parse_args()


@torch.no_grad()
def banana_rate(model, dl, patch, area_range, rot, seed=0):
    """(top-1 = banana 비율, banana 평균 확률). patch=None이면 원본 이미지."""
    gen = torch.Generator().manual_seed(seed)
    hits = probs = total = 0
    for x, _ in dl:
        x = x.to(DEVICE)
        tf = sample_transforms(x.size(0), area_range, rot, generator=gen)  # 패치 유무와 관계없이 같은 난수 소비
        if patch is not None:
            x = apply_patch(x, patch.to(DEVICE), tf)
        with torch.autocast("cuda", enabled=DEVICE == "cuda"):
            logits = model(x).float()
        hits += (logits.argmax(1) == TARGET_CLASS).sum().item()
        probs += logits.softmax(1)[:, TARGET_CLASS].sum().item()
        total += x.size(0)
    return hits / total, probs / total


def plot_transfer(results, n_images, rot, path):
    model_names = list(results["models"])
    conds = list(next(iter(results["models"].values())))
    colors = ["#bbbbbb", "#7f7f7f", "#e8b400"]
    xs, w = np.arange(len(model_names)), 0.26
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, c in enumerate(conds):
        vals = [results["models"][m][c]["rate"] for m in model_names]
        bars = ax.bar(xs + (i - 1) * w, vals, w, label=c, color=colors[i])
        ax.bar_label(bars, labels=[f"{v:.0%}" for v in vals], fontsize=8, padding=2)
    ax.axhline(0.8, color="red", lw=.8, ls=":")
    ax.set_xticks(xs, [m + ("\n(attacked)" if m == "ResNet-50" else "\n(transfer)") for m in model_names])
    ax.set_ylabel("classified as banana")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"{n_images} held-out images | patch area 10-20% | rotation ±{rot:g}° | random position/brightness",
                 fontsize=10)
    ax.legend()
    ax.grid(axis="y", alpha=.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_area_curve(curve, path):
    a = [x * 100 for x in curve["areas"]]
    plt.figure(figsize=(6, 4))
    plt.plot(a, curve["Our patch"], "o-", label="Our patch")
    plt.plot(a, curve["Random noise patch"], "s--", color="gray", label="Random noise patch")
    plt.axvspan(10, 20, color="gold", alpha=.15, label="training range (10-20%)")
    plt.axhline(0.8, color="red", lw=.8, ls=":", label="goal 80%")
    plt.xlabel("patch area (% of image)")
    plt.ylabel("classified as banana (ResNet-50)")
    plt.ylim(-0.02, 1.02)
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


@torch.no_grad()
def plot_examples(model, ds, patch, rot, path, k=6):
    x = torch.stack([ds[i][0] for i in range(0, 6 * k, 6)])  # 6장 간격으로 골라 다양한 클래스가 나오게
    tf = sample_transforms(k, (0.15, 0.15), rot, generator=torch.Generator().manual_seed(7))
    adv = apply_patch(x, patch, tf)

    def top1(imgs):
        p = model(imgs.to(DEVICE)).float().softmax(1).cpu()
        conf, idx = p.max(1)
        return [(LABELS[i], c) for i, c in zip(idx.tolist(), conf.tolist())]

    rows = [("Before", x, top1(x)), ("After", adv, top1(adv))]
    fig, ax = plt.subplots(2, k, figsize=(3 * k, 6.6))
    for r, (row_name, imgs, preds) in enumerate(rows):
        for j in range(k):
            label, conf = preds[j]
            ax[r, j].imshow(imgs[j].permute(1, 2, 0))
            ax[r, j].axis("off")
            ax[r, j].set_title(f"{label}\n{conf:.0%}", fontsize=10, color="#b8860b" if label == "banana" else "black")
        ax[r, 0].text(-20, IMG_SIZE / 2, row_name, rotation=90, va="center", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def area_curve(model, dl, patch, noise, rot):
    """패치 면적을 고정해 가며 ResNet-50 성공률 측정 (우리 패치 vs 랜덤 노이즈)."""
    curve = {"areas": CURVE_AREAS, "Our patch": [], "Random noise patch": []}
    for a in CURVE_AREAS:
        ours = banana_rate(model, dl, patch, (a, a), rot)[0]
        rand = banana_rate(model, dl, noise, (a, a), rot)[0]
        curve["Our patch"].append(ours)
        curve["Random noise patch"].append(rand)
        print(f"  area {a:5.1%}: ours {ours:6.1%} | noise {rand:6.1%}", flush=True)
    return curve


def main():
    args = parse_args()
    patch = torch.load(args.patch).cpu()
    noise = torch.rand(patch.shape, generator=torch.Generator().manual_seed(123))  # 같은 크기 랜덤 노이즈 대조군
    ds = eval_dataset(args.images)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
    conditions = {"No patch": None, "Random noise patch": noise, "Our patch": patch}
    results = {"images": len(ds), "rot_deg": args.rot, "area_range": list(AREA_RANGE), "models": {}}
    print(f"device: {DEVICE} | eval images: {len(ds)} (val split, not used for training)")

    for name in args.models:
        model = load_model(name)
        row = {}
        for cname, p in conditions.items():
            rate, prob = banana_rate(model, dl, p, AREA_RANGE, args.rot)
            row[cname] = {"rate": rate, "mean_prob": prob}
            print(f"{name:13s} | {cname:18s} | banana {rate:6.1%} | mean p(banana) {prob:.3f}", flush=True)
        results["models"][name] = row

        if name == "ResNet-50":  # 공격 대상 모델만 추가 분석
            results["area_curve"] = area_curve(model, dl, patch, noise, args.rot)
            plot_area_curve(results["area_curve"], f"{args.out}/area_curve.png")
            plot_examples(model, ds, patch, args.rot, f"{args.out}/examples.png")
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    with open(f"{args.out}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    plot_transfer(results, len(ds), args.rot, f"{args.out}/transfer.png")

    if "ResNet-50" in results["models"]:
        r = results["models"]["ResNet-50"]["Our patch"]["rate"]
        print(f"\n완료 기준 (ResNet-50, 무작위 변형, >= 80%): {r:.1%} -> {'달성' if r >= 0.8 else '미달'}")


if __name__ == "__main__":
    main()
