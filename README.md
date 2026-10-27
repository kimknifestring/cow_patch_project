# ResNet-50을 "바나나"로 속이는 Adversarial Patch

이미지 어디에 붙여도 ResNet-50이 **banana(ImageNet 954번)** 라고 답하게 만드는 패치를 학습.
Brown et al. 2017, _Adversarial Patch_ 를 재현하는 실험.

## 실행 방법

```bash
pip install -r requirements.txt

# 1) 패치 학습
python train.py

# 2) 평가 (학습에 안 쓴 이미지 1000장 사용)
python evaluate.py
```

결과는 모두 `outputs/` 폴더에 저장됩니다.

| 파일                          | 내용                                                     |
| ----------------------------- | -------------------------------------------------------- |
| `banana_patch.png`            | 학습된 패치                                              |
| `banana_patch_print_1024.png` | 인쇄/웹캠 테스트용 큰 패치                               |
| `training.png`                | 학습 곡선                                                |
| `transfer.png`                | 패치 없음 / 랜덤 노이즈 패치 / 우리 패치 × 4개 모델 비교 |
| `area_curve.png`              | 패치 크기에 따른 성공률                                  |
| `examples.png`                | 패치 붙이기 전/후 예측 예시                              |
| `results.json`                | 위 숫자 전부                                             |

## 하이퍼파라미터

```bash
python train.py --steps 3000              # 스텝 수
python train.py --batch-size 16           # 배치 사이즈
python train.py --area 0.05 0.10          # 패치 크기
```

전체 옵션은 `python train.py -h`, `python evaluate.py -h`.

## 코드 구조

```
train.py                 패치 학습
evaluate.py              평가 + 그래프
patch_attack/common.py   데이터 · 모델 불러오기
```
