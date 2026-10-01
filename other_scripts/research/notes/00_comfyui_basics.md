# ComfyUI 기본기 — 캐릭터 분리 연구로 가는 길

## 1. 핵심 개념: 데이터 타입이 곧 연결 규칙

ComfyUI는 노드 그래프다. 어떤 소켓끼리 연결되는지는 **타입**으로 결정된다.
이 타입 5개만 이해하면 그래프 90%가 읽힌다.

| 타입 | 정체 | 어디서 나오나 |
|---|---|---|
| `MODEL` | UNet (노이즈 예측기) | `CheckpointLoaderSimple` 출력 0 |
| `CLIP` | 텍스트 인코더 | 출력 1 |
| `VAE` | Latent ↔ 픽셀 변환기 | 출력 2 |
| `CONDITIONING` | 인코딩된 프롬프트 | `CLIPTextEncode` |
| `LATENT` | 잠재공간 이미지 (1/8 해상도) | `EmptyLatentImage`, `KSampler` |

`IMAGE`(픽셀)와 `LATENT`(잠재)는 **다른 타입**이다. 헷갈리면 그래프가 안 붙는다.
`MASK`도 별개 타입 — 캐릭터 분리에서 주역이 된다.

## 2. 최소 txt2img 그래프

```
CheckpointLoaderSimple
   ├─MODEL──────────────────────────┐
   ├─CLIP─┬─ CLIPTextEncode(pos) ─CONDITIONING─┐
   │      └─ CLIPTextEncode(neg) ─CONDITIONING─┤
   └─VAE────────────────────┐                  │
                            │      EmptyLatentImage ─LATENT─┐
                            │                  │            │
                            │              KSampler ◄───────┘
                            │                  │
                            │               LATENT
                            └──────► VAEDecode
                                          │
                                        IMAGE → SaveImage
```

`research/workflows/01_sdxl_txt2img.api.json` 이 정확히 이 그래프다.
ComfyUI 웹UI에서 **Workflow > Open** 으로 불러올 수 있다.

## 3. KSampler 파라미터 — 실제로 중요한 것만

| 파라미터 | 의미 | Illustrious 권장 |
|---|---|---|
| `steps` | 디노이즈 반복 횟수 | 24~30 |
| `cfg` | 프롬프트 강제력. 높으면 과포화/붕괴 | **5~7** (SD1.5보다 낮게) |
| `sampler_name` | 디노이즈 알고리즘 | `euler_ancestral` |
| `scheduler` | 스텝별 노이즈 감소 곡선 | `normal` 또는 `karras` |
| `denoise` | 1.0=완전 생성, <1.0=img2img 강도 | txt2img는 1.0 |
| `seed` | 난수 시드. **고정해야 비교 실험이 가능** | 실험 시 고정 |

> 연구 노트: 분리 기법을 비교할 때 `seed`, `steps`, `cfg`, `sampler`를 전부
> 고정하고 프롬프트/컨디셔닝만 바꿔야 원인이 분리된다.

## 4. Illustrious(애니 SDXL) 프롬프트 규칙

Danbooru 태그 기반이다. 자연어 문장보다 **콤마 구분 태그**가 잘 듣는다.

```
masterpiece, best quality, very aesthetic, absurdres,   ← 품질 태그 (앞)
1girl, long red hair, blue dress,                        ← 피주체 태그
standing, simple background                              ← 구도/배경
```

- 해상도는 SDXL 기준 총 픽셀 ~1024² 유지. `1024x1024`, `832x1216`, `1216x832`.
- 512x512로 내리면 SDXL은 품질이 무너진다 (SD1.5와 다른 점).

## 5. 다중 캐릭터 = 여기서 문제가 시작된다

`2girls, one girl with long red hair and blue dress, another girl with short
black hair and white shirt` 같은 프롬프트를 넣으면 거의 확실히:

- **속성 누출 (attribute bleeding)** — 빨간 머리가 두 명 다 되거나, 파란 드레스를
  검은 머리 쪽이 입는다.
- CLIP은 토큰을 전역으로 섞어 어텐션하므로 "누가 무엇을 입었는지"를 공간적으로
  구분하지 못한다. 이게 캐릭터 분리 연구의 근본 원인이다.

`01_sdxl_txt2img.api.json` 의 프롬프트가 의도적으로 이 케이스다.
**먼저 문제를 눈으로 확인**하고, 그 다음에 해법으로 넘어간다.

### 해법 계보 (다음 단계에서 하나씩)

| 접근 | 방식 | ComfyUI 수단 |
|---|---|---|
| 영역 분할 컨디셔닝 | 캔버스를 영역으로 쪼개 각 영역에 다른 프롬프트 | `ConditioningSetMask` + `ConditioningCombine` (**내장**) |
| Latent Couple | 영역별 latent 를 따로 디노이즈 후 합성 | 커스텀 노드 |
| ControlNet 포즈 고정 | OpenPose로 인물 배치를 먼저 고정 | `controlnet/` 모델 필요 |
| 개별 생성 후 합성 | 캐릭터를 따로 만들고 마스크로 합침 | `ImageCompositeMasked` (**내장**) |
| 사후 인스턴스 분리 | 생성된 이미지를 SAM3로 캐릭터별 분리 | `SAM3_Detect` (**내장**) |

`ConditioningSetMask`가 내장이라 **커스텀 노드 없이 영역 분할 컨디셔닝까지
바로 실험 가능**하다. 이게 다음 단계의 출발점.

## 6. 8GB VRAM 실전 수칙

- SDXL 1024² 1장: 약 6~7GB 사용. 배치 2 이상은 위험.
- 모델 바꾸면 이전 모델이 VRAM에 남아 있을 수 있다 →
  Manager 의 **Unload Models** 또는 `/free` 엔드포인트로 비운다.
- ControlNet/SAM을 동시에 물리면 `run_comfyui_lowvram.bat` 사용.
- 브라우저 탭도 VRAM을 먹는다. 생성 중 다른 탭 정리.

## 7. 자주 쓰는 API 엔드포인트 (스크립트 실험용)

```
GET  /system_stats          VRAM/버전 확인
GET  /object_info           전체 노드 스펙 (945개)
GET  /object_info/KSampler  단일 노드 스펙
POST /prompt                워크플로우 큐잉  {"prompt": <api json>}
GET  /history/<prompt_id>   결과 조회
POST /free                  {"unload_models":true,"free_memory":true}
```

배치 실험은 웹UI 클릭보다 `POST /prompt` 스크립트가 압도적으로 빠르다.
시드/CFG 스윕 같은 건 파이썬으로 돌리는 게 맞다.

## 8. 영역 분할 컨디셔닝 — 가장 쉬운 첫 해법 (검증됨: 내장 노드)

마스크를 그릴 필요도 없다. `ConditioningSetAreaPercentage`는 0~1 비율로
사각 영역을 지정한다.

```
CLIPTextEncode("1girl, long red hair, blue dress")
   → ConditioningSetAreaPercentage(x=0.0, y=0, width=0.5, height=1.0, strength=1.0) ─┐
                                                                                     ├→ ConditioningCombine → KSampler.positive
CLIPTextEncode("1girl, short black hair, white shirt")                                │
   → ConditioningSetAreaPercentage(x=0.5, y=0, width=0.5, height=1.0, strength=1.0) ─┘
```

여기에 전역 배경/품질 프롬프트를 세 번째 컨디셔닝(영역 지정 없이)으로 더해
`ConditioningCombine`을 중첩하면 화면 전체 일관성이 잡힌다.

확인된 관련 내장 노드:
`ConditioningSetMask`(mask, strength, set_cond_area), `ConditioningCombine`,
`ConditioningSetArea`, `ConditioningSetAreaPercentage`, `ConditioningConcat`,
`ConditioningAverage`.

- `SetArea*` = 사각형 영역. 빠르고 쉽다. 경계가 직선이라 티가 날 수 있다.
- `SetMask` = 임의 형태 마스크. `SolidMask`/`MaskComposite`로 만들거나
  SAM3 마스크를 재활용. 정밀하지만 손이 더 간다.

> 실험 설계: 동일 seed 로 (a) 단일 프롬프트, (b) SetAreaPercentage 분할,
> (c) SetMask 분할 3종을 돌려 속성 누출이 얼마나 줄어드는지 비교.
