# 실험 01 — 베이스라인 속성 누출 재현

**날짜**: 2026-09-12
**목적**: 다중 캐릭터 프롬프트에서 속성 누출이 실제로 발생하는지 확인 (해법 없이)

## 설정

| 항목 | 값 |
|---|---|
| 워크플로우 | `research/workflows/01_sdxl_txt2img.api.json` |
| 체크포인트 | Illustrious-XL-v0.1.safetensors |
| 해상도 | 1024 x 1024 |
| seed | 20260912 (고정) |
| steps / cfg | 28 / 6.0 |
| sampler / scheduler | euler_ancestral / normal |
| 속도 | 약 1.6 it/s (~18초/장, RTX 3070) |
| VRAM 사용 | 약 4.9 GB |

### 프롬프트 (positive)

```
masterpiece, best quality, very aesthetic, absurdres, 2girls,
one girl with long red hair and blue dress,
another girl with short black hair and white shirt,
standing side by side, simple background
```

## 결과

출력: `ComfyUI/output/baseline_2girls_00001_.png`

| 의도 | 실제 생성 | 판정 |
|---|---|---|
| 빨간 머리 → 파란 드레스 | 빨간 머리 → **흰 셔츠** | 누출 |
| 검은 머리 → 흰 셔츠 | 검은 머리 → **파란 드레스** | 누출 |
| 머리 길이: red=long, black=short | red=중간, black=long | 부분 실패 |
| 인원 수 2명 | 2명 | 성공 |
| 나란히 서기 | 성공 | 성공 |

### 관찰

1. **의상 속성이 완전히 교환됐다.** 두 속성이 뭉개진 게 아니라 서로 자리를
   바꿨다 — CLIP이 "red hair"와 "blue dress"를 같은 인물에 묶는 바인딩 정보를
   잃고, 두 속성 집합을 임의로 재배치했다는 뜻이다.
2. **머리 길이 태그도 무시됐다.** `long red hair` / `short black hair` 중
   short 쪽이 long 으로 생성됐다. 색상보다 길이 속성이 더 약하게 걸린다.
3. **색상 자체는 둘 다 살아남았다.** 팔레트(빨강/검정/파랑/흰색)는 전부
   등장했다. 즉 모델은 "무엇이 있는지"는 알지만 "누구의 것인지"를 모른다.

> 이것이 캐릭터 분리 연구의 핵심 문제 정의다:
> **속성의 존재(what)가 아니라 속성의 귀속(whose)이 실패한다.**

## 다음 실험

- **02**: 동일 seed + `ConditioningSetAreaPercentage` 좌/우 분할 → 누출 해소 여부
- **03**: 동일 seed + `ConditioningSetMask` 임의 형태 마스크 → 경계 자연스러움 비교
- **04**: 생성물에 `SAM3_Detect(individual_masks=True)` 적용 → 사후 인스턴스 분리

seed / steps / cfg / sampler 는 전 실험 고정. 바꾸는 건 컨디셔닝 구조뿐.
