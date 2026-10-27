# ResNet-50을 "바나나"로 속이는 Adversarial Patch

이미지 어디에 붙여도 ResNet-50이 **banana(ImageNet 954번)** 라고 답하게 만드는 패치를 학습합니다.
Brown et al. 2017, *Adversarial Patch* 방식을 디지털 환경에서 재현하는 단계입니다.

## 실행 방법

```bash
pip install -r requirements.txt

# 1) 패치 학습 (GPU 기준 10~20분, 학습용 배경 이미지는 처음 실행 때 자동 다운로드 ~95MB)
python train.py

# 2) 평가 (학습에 안 쓴 이미지 1000장 + 다른 모델 전이율)
python evaluate.py
```

결과는 모두 `outputs/` 폴더에 저장됩니다.

| 파일 | 내용 |
|---|---|
| `banana_patch.png` | 학습된 패치 |
| `banana_patch_print_1024.png` | 인쇄/웹캠 테스트용 큰 패치 |
| `training.png` | 학습 곡선 |
| `transfer.png` | 패치 없음 / 랜덤 노이즈 패치 / 우리 패치 × 4개 모델 비교 |
| `area_curve.png` | 패치 크기에 따른 성공률 |
| `examples.png` | 패치 붙이기 전/후 예측 예시 |
| `results.json` | 위 숫자 전부 |

**완료 기준**: `evaluate.py` 마지막 줄의 ResNet-50 성공률이 80% 이상.

## 어떻게 동작하나

1. **배경 이미지**: [Imagenette](https://github.com/fastai/imagenette) (ImageNet 10개 클래스). banana가 없어서 "패치 없으면 banana ≈ 0%"인 깨끗한 기준선이 됩니다. `train` 폴더로 학습, `val` 폴더로 평가합니다.
2. **패치**: 무작위 노이즈로 시작하는 128×128 정사각형. 모델 가중치는 고정하고 패치 픽셀만 학습합니다.
3. **EOT (무작위 변형)**: 매 이미지마다 패치를 다르게 변형해서 붙입니다.
   - 크기: 이미지 면적의 10~20%
   - 회전: ±45°
   - 위치: 이미지 안 아무 곳
   - 밝기 0.75~1.25배, 대비 0.8~1.2배
4. **Loss**: `-log p(banana)` + TV loss(패치를 매끄럽게 해서 나중에 인쇄할 때 유리하게)
5. **평가**: 같은 무작위 변형을 세 조건에 똑같이 적용해 비교합니다.
   - 패치 없음 / 같은 크기의 랜덤 노이즈 패치 / 우리 패치
   - ResNet-50(공격 대상) + VGG-16, MobileNet-V2, DenseNet-121(전이, 추론만)

## 자주 바꿀 옵션

```bash
python train.py --steps 3000              # 성공률이 80% 미만이면 더 오래 학습
python train.py --batch-size 16           # GPU 메모리가 부족할 때
python train.py --area 0.05 0.10          # 더 작은 패치로 도전
python evaluate.py --models ResNet-50     # 전이 측정 생략하고 빠르게
```

전체 옵션은 `python train.py -h`, `python evaluate.py -h`.

## 코드 구조

```
train.py                 패치 학습
evaluate.py              평가 + 그래프
patch_attack/common.py   데이터 · 모델 불러오기 · EOT 합성(apply_patch) · TV loss
```

## 이 단계의 범위

- 분류 모델(ResNet)은 "이미지 전체가 무엇인가"만 답합니다. 그래서 데모는 **"화면 전체가 banana로 분류된다"** 형태입니다.
- 사람 박스를 지우는 장면(YOLO, Thys et al. 2019), NPS 기반 인쇄 보정은 다음 단계입니다.
