# ResNet-PyTorch-Implementation

PyTorch를 활용하여 ResNet v2를 직접 구현하고 실험한 저장소입니다.
본 프로젝트에서는 ResNet v2의 핵심 구조인 BasicBlock, Bottleneck, Projection Shortcut을 직접 구현했으며,
uv를 패키지 매니저로 사용하여 실험 환경을 관리합니다.

---

본 구현에 대한 분석은 아래 블로그에 기록해 두었습니다.

<p align="center">
  <a href="https://velog.io/@jeongjae/ResNetV2-PyTorch-Implementation">
    <img src="./pinggu_circle.png" width="150" height="150">
  </a>
</p>

<p align="center">
  <a href="https://velog.io/@jeongjae/ResNetV2-PyTorch-Implementation">
    <b>ResNet v2 논문 리뷰 및 구현 by J2-J2</b>
  </a>
</p>
---

## Environment Setup
```bash
# 가상환경 활성화 및 패키지 동기화
uv sync
```

## Project Structure
├── configs/          # 모델 및 학습 하이퍼파라미터 설정 (YAML)
├── data_imagenette/  # Imagenette 데이터셋 저장 경로
├── models/           # ResNet v2 (BasicBlock, Bottleneck) 코드 구현
├── notebooks/        # Colab 가속 학습 및 프로토타입 검증 환경
├── results/          # 학습 완료된 가중치(.pth) 및 실험 평가지표 기록
├── inference.py      # 단일 이미지 추론(Inference) 파이프라인 스크립트
└── main.py           # 로컬 환경 전체 학습 제어 파이프라인 스크립트

## Training
```bash
# ResNet-PyTorch-Implementation 폴더에서 실행할 것
python main.py
```

## Inference
```bash
# ResNet-PyTorch-Implementation 폴더에서 실행할 것
python inference.py --image_path <path_to_image>
```
