# Stable Diffusion 캐릭터 분리 연구

## 환경

| 항목 | 값 |
|---|---|
| GPU | NVIDIA RTX 3070 (8GB VRAM, sm_86) |
| 드라이버 | 591.86 (CUDA 13.1) |
| Python | 3.10.11 (venv: `.venv`) |
| PyTorch | cu128 빌드 |
| ComfyUI | 소스 클론 (`ComfyUI/`) |

Python 3.10을 선택한 이유: 캐릭터/인물 분리에 쓰이는 커스텀 노드 의존성
(insightface, onnxruntime, ultralytics, segment-anything)이 Windows에서
cp310 prebuilt wheel 지원이 가장 안정적임.

## 실행

```
run_comfyui.bat            # 기본 (http://127.0.0.1:8188)
run_comfyui.bat --lowvram    # SDXL + 세그멘테이션 모델 동시 로드 시
```

## 디렉터리

```
Work/
|-- ComfyUI/             application, custom nodes, input/output
|   |-- extra_model_paths.yaml
|-- models/              checkpoints, LoRAs, VAE
|-- .venv/               Python environment
|-- run_comfyui.bat
|-- other_scripts/       research notes and workflows
```

## 모델 배치 위치

| 종류 | 경로 |
|---|---|
| Checkpoint (SD1.5/SDXL) | `models/checkpoints/` |
| Diffusion model (Anima/Flux 등 UNet 단독) | `models/diffusion_models/` |
| VAE | `models/vae/` |
| LoRA | `models/loras/` |
| ControlNet | `models/controlnet/` |
| CLIP / Text encoder | `models/clip/`, `models/text_encoders/` |
| CLIP Vision | `models/clip_vision/` |
| **배경 제거 (RMBG/BiRefNet)** | `models/background_removal/` |
| **객체 검출 (YOLO 등)** | `models/detection/` |
| Upscaler | `models/upscale_models/` |
| Embedding (Textual Inversion) | `models/embeddings/` |

> `background_removal`, `detection` 은 이 ComfyUI 버전이 기본 내장한 폴더로,
> 캐릭터 분리 파이프라인에서 곧바로 쓰인다. SAM/SAM2 기반 커스텀 노드를
> 추가하면 노드별 자체 경로(`models/sams` 등)가 생성된다.

## 캐릭터 분리에 쓸 수 있는 내장 노드 (커스텀 노드 불필요)

첫 기동 시 **945개 노드** 등록 확인. 커스텀 노드 없이 베이스라인 구성 가능.

| 노드 | 역할 |
|---|---|
| `SAM3_Detect` | 텍스트 conditioning 기반 세그멘테이션. `individual_masks=True`로 **인스턴스별 마스크 분리** — 캐릭터 분리의 핵심 |
| `SAM3_TrackToMask` | 시퀀스/비디오에서 특정 객체 인덱스 추적 → 마스크 |
| `RTDETR_detect` | 클래스 지정 객체 검출 → bbox (SAM3의 `bboxes` 입력으로 연결) |
| `LoadBackgroundRemovalModel` + `RemoveBackground` | 배경 제거 → 단일 마스크 |
| `MediaPipeFaceMask` | 얼굴 영역 마스크 (캐릭터 식별 보조) |
| `GrowMask` / `FeatherMask` / `ThresholdMask` | 마스크 경계 후처리 |
| `MaskComposite` / `InvertMask` | 마스크 논리 연산 (오클루전 처리) |
| `ImageCropToMask` / `CropByBBoxes` | 분리된 영역 추출 |
| `ImageCompositeMasked` | 재합성 검증 |

### 베이스라인 파이프라인 구상

```
LoadImage
  → RTDETR_detect (class=person) ──bboxes──┐
  → SAM3_Detect (individual_masks=True) ◄──┘
      → masks (N개)
      → GrowMask/FeatherMask (경계 정리)
      → ImageCropToMask → 캐릭터별 분리 이미지
      → ImageCompositeMasked (재합성 검증)
```

필요 모델: SAM3 체크포인트, RT-DETR 가중치, 배경제거 모델(RMBG/BiRefNet).
→ `LoadBackgroundRemovalModel` 콤보는 현재 비어 있음 (모델 미배치).

## Anima Turbo v1.1 구성

SDXL이 아니라 **DiT 아키텍처**다. 체크포인트 하나로 끝나지 않고 확산모델/텍스트인코더/VAE를
따로 올려야 한다. Illustrious·Pony용 LoRA는 호환되지 않는다.

| 역할 | 파일 | 경로 |
|---|---|---|
| 확산 모델 (28 blocks, bf16) | `anima_turboV11.safetensors` (3.9GB) | `models/diffusion_models/` |
| 그림체 LoRA (UNet 전용) | `Anima_Style_Lora.safetensors` (132MB) | `models/loras/` |
| 텍스트 인코더 (Qwen3-0.6B) | `qwen_3_06b_base.safetensors` (1.2GB) | `models/text_encoders/` |
| VAE (16ch, Wan21 latent) | `qwen_image_vae.safetensors` (243MB) | `models/vae/` |

- 노드: `UNETLoader` → `LoraLoaderModelOnly` → `KSampler`,
  텍스트는 `CLIPLoader`(type=`stable_diffusion`, Qwen3-0.6B 자동 인식) → `CLIPTextEncode`
- **turbo 샘플링**: 8 steps / CFG 1.0 / `euler` + `simple` (non-turbo는 30 steps / CFG 4)
- 프롬프트는 **부루 태그가 아니라 자연어**. Qwen3 + T5 이중 토크나이즈이고
  T5 임베딩 테이블은 확산모델의 `llm_adapter`에 내장돼 별도 T5 파일이 필요 없다.
- 측정: 1024², 8 steps, 3.7초 (2.15 it/s), RTX 3070 8GB에서 `--lowvram` 없이 동작

## 8GB VRAM 가이드

- SD1.5 기반: 여유 있음. 512~768px 권장.
- SDXL: 가능하나 세그멘테이션 모델 병행 시 `--lowvram` 필요.
- Flux: 8GB에서는 GGUF Q4 양자화 + `--lowvram` 조합만 현실적.

## 진행 상황

- [x] ComfyUI + venv + PyTorch cu128 설치, CUDA 검증
- [x] ComfyUI-Manager 설치
- [x] 베이스 체크포인트: **Illustrious-XL v0.1** (애니/일러스트 SDXL, 6.5GB)
- [x] 첫 생성 성공 — 1024², 1.6 it/s, VRAM 4.9GB
- [x] 실험 01: 속성 누출 재현 (`other_scripts/research/notes/01_baseline_attribute_bleeding.md`)
- [x] **Anima Turbo v1.1 + 그림체 LoRA 도입** (2026-09-14)
      — `other_scripts/research/workflows/02_anima_turbo_txt2img.api.json`, 1024² / 8 steps / 3.7초
- [ ] 실험 02: `ConditioningSetAreaPercentage` 영역 분할
- [ ] 실험 03: `ConditioningSetMask` 마스크 분할
- [ ] 실험 04: `SAM3_Detect` 사후 인스턴스 분리 (SAM3 가중치 필요)

### 연구 방향

**생성 단계에서의 캐릭터 격리**가 목표. 현 단계는 ComfyUI 기본기 습득.
문제 정의는 실험 01에서 확립됨 — 속성의 *존재*가 아니라 *귀속*이 실패한다.

### 읽을 순서

1. `other_scripts/research/notes/00_comfyui_basics.md` — 노드 타입, KSampler, 프롬프트 규칙
2. `other_scripts/research/notes/01_baseline_attribute_bleeding.md` — 베이스라인 실험 결과
3. `other_scripts/research/workflows/01_sdxl_txt2img.api.json` — 웹UI에서 Open 으로 로드

## 설치 검증 결과 (2026-09-12)

- `main.py --quick-test-for-ci` 정상 종료, 커스텀 노드 임포트 오류 없음
- ComfyUI 0.35.0 / frontend 1.52.7 / ComfyUI-Manager V3.41
- `GET /system_stats` → HTTP 200, device `cuda:0 RTX 3070`, VRAM free 7.44GB
- `GET /object_info` → 945 nodes
- triton 미설치 (Windows 기본): `torch.compile` / SageAttention 미사용.
  필요 시 `uv pip install triton-windows` — 필수 아님.
