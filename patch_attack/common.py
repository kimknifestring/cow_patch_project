"""데이터, 모델, EOT 합성 등 학습/평가에서 같이 쓰는 코드."""
import math
import os
import tarfile
import urllib.request

import torch
import torch.nn.functional as F
import torchvision as tv
from torchvision import transforms as T

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_CLASS = 954  # ImageNet 954 = banana
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LABELS = tv.models.ResNet50_Weights.IMAGENET1K_V2.meta["categories"]

# EOT 기본값
AREA_RANGE = (0.10, 0.20)   # 패치가 차지하는 이미지 면적 비율
ROT_DEG = 45                # 회전 ±45°
BRIGHTNESS = (0.75, 1.25)   # 패치 밝기 배율
CONTRAST = (0.8, 1.2)       # 패치 대비 배율

# ---------------------------------------------------------------- data

DATA_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"


def ensure_imagenette(root="data"):
    """Imagenette(ImageNet 10개 클래스 부분집합, 160px, 약 95MB)를 받아서 경로를 돌려준다.
    10개 클래스에 banana가 없어서 '패치 없을 때 banana ≈ 0%'인 깨끗한 기준선이 된다."""
    path = os.path.join(root, "imagenette2-160")
    if not os.path.isdir(path):
        os.makedirs(root, exist_ok=True)
        tgz = os.path.join(root, "imagenette2-160.tgz")
        print("downloading Imagenette ...")
        urllib.request.urlretrieve(DATA_URL, tgz)
        with tarfile.open(tgz) as tar:
            tar.extractall(root)
    return path


def train_dataset(root="data"):
    tf = T.Compose([T.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)), T.RandomHorizontalFlip(), T.ToTensor()])
    return tv.datasets.ImageFolder(os.path.join(ensure_imagenette(root), "train"), tf)


def eval_dataset(n_images, seed=0, root="data"):
    """학습에 쓰지 않은 val 폴더에서 n장을 고정 시드로 뽑는다."""
    tf = T.Compose([T.Resize(IMG_SIZE), T.CenterCrop(IMG_SIZE), T.ToTensor()])
    full = tv.datasets.ImageFolder(os.path.join(ensure_imagenette(root), "val"), tf)
    idx = torch.randperm(len(full), generator=torch.Generator().manual_seed(seed))[:n_images].tolist()
    return torch.utils.data.Subset(full, idx)

# ---------------------------------------------------------------- models


class Normalized(torch.nn.Module):
    """ImageNet 정규화를 모델 앞에 붙여서 이미지/패치는 항상 [0, 1] 픽셀 공간에서 다룬다."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


MODELS = {
    "ResNet-50":    lambda: tv.models.resnet50(weights="IMAGENET1K_V2"),
    "VGG-16":       lambda: tv.models.vgg16(weights="IMAGENET1K_V1"),
    "MobileNet-V2": lambda: tv.models.mobilenet_v2(weights="IMAGENET1K_V2"),
    "DenseNet-121": lambda: tv.models.densenet121(weights="IMAGENET1K_V1"),
}


def load_model(name):
    m = Normalized(MODELS[name]()).eval().to(DEVICE)
    for p in m.parameters():
        p.requires_grad_(False)
    return m

# ---------------------------------------------------------------- EOT


def sample_transforms(n, area_range=AREA_RANGE, rot_deg=ROT_DEG, generator=None):
    """이미지 n장 각각에 대한 무작위 (크기, 회전, 위치, 밝기, 대비).
    난수를 뽑는 순서를 바꾸면 같은 시드라도 평가 결과가 달라진다."""
    def uniform(lo, hi):
        return lo + (hi - lo) * torch.rand(n, generator=generator)

    side = uniform(*area_range).sqrt()  # 이미지 한 변 대비 패치 한 변 비율
    theta = uniform(-1, 1) * math.radians(rot_deg)

    # 회전한 정사각형의 외접 반폭 (정규화 좌표, 이미지 반폭 = 1) -> 패치가 이미지 밖으로 안 나가게
    half_extent = side * (theta.cos().abs() + theta.sin().abs())
    max_shift = (1 - half_extent).clamp(min=0)
    tx = uniform(-1, 1) * max_shift
    ty = uniform(-1, 1) * max_shift

    bright = uniform(*BRIGHTNESS)
    contr = uniform(*CONTRAST)
    return dict(side=side, theta=theta, tx=tx, ty=ty, bright=bright, contr=contr)


def apply_patch(images, patch, tf):
    """images (N,3,H,W) [0,1] 위에 patch (3,P,P) [0,1]를 tf대로 변형해서 합성한다.
    affine_grid + grid_sample이라 패치까지 gradient가 그대로 흐른다."""
    n, _, h, w = images.shape
    dev = images.device
    side, theta = tf["side"].to(dev), tf["theta"].to(dev)
    tx, ty = tf["tx"].to(dev), tf["ty"].to(dev)

    # 출력 이미지 좌표 x -> 패치 좌표 u = R(-θ)(x - t) / side
    cos, sin = theta.cos() / side, theta.sin() / side
    A = torch.stack([torch.stack([cos, sin], 1), torch.stack([-sin, cos], 1)], 1)  # (N,2,2)
    t = torch.stack([tx, ty], 1).unsqueeze(-1)
    grid = F.affine_grid(torch.cat([A, -A @ t], 2), (n, 3, h, w), align_corners=False)

    # 밝기/대비 (패치에만, 조명 변화 흉내)
    contrast = tf["contr"].to(dev).view(n, 1, 1, 1)
    brightness = tf["bright"].to(dev).view(n, 1, 1, 1)
    p = patch.unsqueeze(0).expand(n, -1, -1, -1)
    mean = p.mean(dim=(1, 2, 3), keepdim=True)
    p = ((p - mean) * contrast + mean) * brightness
    p = p.clamp(0, 1)

    warped = F.grid_sample(p, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    mask = F.grid_sample(torch.ones_like(p[:, :1]), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return images * (1 - mask) + warped * mask


def tv_loss(p):
    """Total variation: 이웃 픽셀 차이의 평균. 패치를 매끄럽게 만들어 인쇄에 유리하게."""
    return (p[:, 1:, :] - p[:, :-1, :]).abs().mean() + (p[:, :, 1:] - p[:, :, :-1]).abs().mean()
