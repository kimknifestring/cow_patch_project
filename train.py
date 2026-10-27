"""ResNet-50을 banana로 속이는 adversarial patch 학습 (Brown et al. 2017).

    python train.py                  # GPU 기본 설정
    python train.py --steps 400 --batch-size 16   # CPU에서 빠르게 확인
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision as tv

from patch_attack.common import (AREA_RANGE, DEVICE, ROT_DEG, TARGET_CLASS, apply_patch,
                                 load_model, sample_transforms, train_dataset, tv_loss)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--patch-px", type=int, default=128, help="패치 파라미터 해상도")
    ap.add_argument("--area", type=float, nargs=2, default=AREA_RANGE, help="패치 면적 비율 범위")
    ap.add_argument("--rot", type=float, default=ROT_DEG, help="회전 범위 ±deg")
    ap.add_argument("--tv-weight", type=float, default=0.1)
    ap.add_argument("--noise-std", type=float, default=0.02, help="합성 후 가우시안 노이즈")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs")
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    print(f"device: {DEVICE} | args: {vars(args)}")

    model = load_model("ResNet-50")
    ds = train_dataset()
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                     num_workers=args.workers, pin_memory=DEVICE == "cuda")
    print(f"train images: {len(ds)}")

    P = args.patch_px
    patch = torch.rand(3, P, P, device=DEVICE, requires_grad=True)  # 무작위 노이즈에서 시작
    init_patch = patch.detach().cpu().clone()
    opt = torch.optim.Adam([patch], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr * 0.1)

    history = dict(step=[], loss=[], success=[])
    log_every = max(args.steps // 30, 10)
    step, t0 = 0, time.time()
    run_loss = run_succ = 0.0
    n_log = 0
    while step < args.steps:
        for x, _ in dl:
            x = x.to(DEVICE, non_blocking=True)
            tf = sample_transforms(x.size(0), args.area, args.rot)
            adv = apply_patch(x, patch, tf)
            adv = (adv + args.noise_std * torch.randn_like(adv)).clamp(0, 1)
            with torch.autocast("cuda", enabled=DEVICE == "cuda"):
                logits = model(adv)
            # banana 확률 최대화 = -log p(banana) 최소화
            loss = -F.log_softmax(logits.float(), 1)[:, TARGET_CLASS].mean() + args.tv_weight * tv_loss(patch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                patch.clamp_(0, 1)

            run_loss += loss.item()
            run_succ += (logits.argmax(1) == TARGET_CLASS).float().mean().item()
            n_log += 1
            step += 1
            if step % log_every == 0 or step == args.steps:
                history["step"].append(step)
                history["loss"].append(run_loss / n_log)
                history["success"].append(run_succ / n_log)
                print(f"step {step:5d}/{args.steps} | loss {run_loss / n_log:.3f} | "
                      f"train banana rate {run_succ / n_log:6.1%} | {time.time() - t0:5.0f}s", flush=True)
                run_loss = run_succ = 0.0
                n_log = 0
            if step >= args.steps:
                break

    final = patch.detach().cpu()
    torch.save(final, f"{args.out}/banana_patch.pt")
    tv.utils.save_image(final, f"{args.out}/banana_patch.png")
    big = F.interpolate(final.unsqueeze(0), size=1024, mode="bilinear", align_corners=False)[0]
    tv.utils.save_image(big, f"{args.out}/banana_patch_print_1024.png")  # 인쇄/웹캠 테스트용
    with open(f"{args.out}/train_history.json", "w") as f:
        json.dump({"args": vars(args), **history}, f, indent=2)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].imshow(init_patch.permute(1, 2, 0)); ax[0].set_title("start: random noise"); ax[0].axis("off")
    ax[1].imshow(final.permute(1, 2, 0)); ax[1].set_title("trained banana patch"); ax[1].axis("off")
    ax[2].plot(history["step"], history["success"], label="banana rate (train batch)")
    ax[2].set_xlabel("step"); ax[2].set_ylim(0, 1); ax[2].grid(alpha=.3)
    ax2 = ax[2].twinx()
    ax2.plot(history["step"], history["loss"], color="tab:orange", label="loss")
    ax[2].set_title("training curve"); ax[2].legend(loc="center right"); ax2.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(f"{args.out}/training.png", dpi=120)
    print(f"saved patch -> {args.out}/banana_patch.png")


if __name__ == "__main__":
    main()
