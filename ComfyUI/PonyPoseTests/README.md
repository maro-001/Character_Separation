# Work — Pony 본 테스트

[모델 출처](https://civitai.com/models/257749/pony-diffusion-v6-xl)의 **V6 (start with this one)**을 `Work/models/checkpoints/ponyDiffusionV6XL_v6StartWithThisOne.safetensors`에 설치하고 제작자 SHA256과 대조했습니다. LoRA는 사용하지 않았습니다. 포즈 모델은 Work의 `models/controlnet`에 설치한 Thibaud/Xinsir OpenPose SDXL입니다. 기존 Character_Separation 워크플로는 유지했습니다.

## 실행과 본 편집

1. http://127.0.0.1:8188 을 열고 Ctrl+F5로 새로고침합니다. 서버가 꺼져 있으면 ComfyUI의 `start_pony_work.bat`를 실행합니다.
2. [Pony_Pose_Work.json](Pony_Pose_Work.json)을 화면에 드래그하거나 워크플로 목록에서 `Pony_Pose_Work`를 엽니다.
3. 본 편집기 노드 40을 우클릭하고 `Open in Openpose Editor`를 선택합니다.
4. 관절을 수정한 뒤 `ControlNet에 자세 보내기` / `Send pose to ControlNet`를 누르고 Run/실행합니다. 본 미리보기는 실행 후 갱신됩니다.
5. 워크플로를 저장하면 수정 본도 함께 저장됩니다. `Reload reference pose`는 해당 워크플로의 기준 본으로 되돌립니다.

기본 파일에는 `D:/Download/pose (1).json` 원본을 넣었습니다. CLIP Skip 2, Euler a 25 steps, CFG 7, normal, 1024×1024, 고정 시드입니다. 추천 ControlNet: **thibaud, strength 0.75, end_percent 0.75**.

[전체 비교 이미지](comparison.html) · [측정 결과](results.csv) · [상세 JSON](results.json) · [모델 다운로드·해시 기록](download_manifest.json). 제어 없음, Thibaud 강도 1/1.5, Xinsir 강도 1.5/2를 원본 본과 두 시드로 비교하고, Thibaud 1.5와 Xinsir 2로 팔을 올린 수정 본을 비교했습니다.

PCK 점수는 몸 관절이 허용 거리 안에 들어온 비율이며 손가락·얼굴 정확도를 보장하지 않습니다. 갑옷·헬멧의 검출 오류도 있을 수 있습니다. 생성 이미지: `Work/ComfyUI/output/PonyPoseTests`. 각 테스트의 UI/API 워크플로는 `workflows` 폴더에 있습니다.

## 생성 중단 시점의 상태

사용자 요청으로 생성과 Work 테스트 서버를 중단했습니다. 완료 이미지 12개, 고강도 Xinsir 수정 본 1개는 시간 초과입니다. 자동 관절 평가는 미완료이며 수치상 최적 설정을 주장하지 않습니다.

고강도 Xinsir 설정은 색과 형태가 무너지는 결과가 나왔습니다. 원본 본의 색 깨짐이 줄어든 Thibaud strength 0.75 / end_percent 0.75 설정을 기본으로 저장했습니다. 이 낮은 강도 설정의 수정 팔 확인은 중단 시점까지 완료하지 못했습니다. 팔 수정 반영을 확인한 Thibaud 1.5 예제는 `Pony_Edited_Arm_Demo.json`에 따로 있습니다.

메모리 부하를 줄이기 위해 기본 워크플로는 VAEDecodeTiled(tile_size 256, overlap 32)를 사용합니다. 생성 시간에는 모델 로드와 디코딩이 포함됩니다. 기존 고강도 결과와 추가 저강도 결과는 디코딩 설정도 달라 엄밀한 동일 조건 비교는 아닙니다.
