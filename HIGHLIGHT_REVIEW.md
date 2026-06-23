# 하이라이트 클립 품질 정리 가이드

새 하이라이트 영상 없이 **기존 115개 클립만** 정리해도 학습 효과가 큽니다.

## 폴더별 체크리스트

### doublekill (더블킬)
- [ ] **DOUBLE KILL** 배너 또는 2연속 킬이 **화면에 보이는** 구간
- [ ] 라운드 시작/전광판만 나오면 **제외** → `review/`
- [ ] 멀티킬(3연속 이상)이면 → `multikill/`로 이동

### multikill (멀티킬)
- [ ] **TRIPLE / MULTI KILL** 배너 또는 3킬 이상
- [ ] 더블만 있으면 → `doublekill/`
- [ ] 가장 적은 클래스(19개) → **애매한 건 과감히 review로**

### save (세이브)
- [ ] 클러치·1vN·시간 끝 직전 역전 등 **세이브 맥락**이 보임
- [ ] 그냥 일반 2킬·3킬이면 → doublekill / multikill

### allkill (올킬)
- [ ] **상대 팀 전멸** 또는 올킬 배너
- [ ] 킬 3~4개만이면 → multikill

### review (검수 보류)
- 애매한 클립은 `clips/review/`로 이동 (학습에서 **자동 제외**)

## 빠른 판별법 (10초)

1. 클립 **중간~끝** 5초만 본다 (앞은 라운드 전환일 수 있음)
2. 킬 배너 / 연속킬 UI / 전멸 장면이 있는지 확인
3. 없으면 `review` 또는 삭제

## 정리 후

```powershell
cd C:\clipAI\files
python scan_clip_folders.py --allow-overwrite
python train_binary.py
python train_highlight_types.py
```
