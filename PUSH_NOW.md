# 🚀 3단계로 GitHub에 올리기 (복사-붙여넣기 가이드)

## ✅ 현재 준비 완료 상태
- Git 초기화 완료
- 코드 파일 커밋 완료 (데이터/모델 제외)
- 안전하게 공개 가능한 상태

---

## STEP 1: GitHub에서 새 레포 만들기 (1분)

1. 브라우저에서 https://github.com/new 열기
2. 아래 정보 입력:
   - **Repository name**: `CrashSeverityNet`
   - **Description**: `Advanced Deep Learning Framework for Traffic Accident Severity Prediction`
   - **Public** 선택 (또는 Private)
   - ⚠️ **체크 해제**: "Add a README file", "Add .gitignore", "Choose a license" 모두 체크 해제!
3. **"Create repository"** 클릭

---

## STEP 2: GitHub가 보여주는 URL 복사 (10초)

레포 생성 후, GitHub가 보여주는 페이지에서:
- "…or push an existing repository from the command line" 섹션 찾기
- 거기 나오는 URL 복사 (예: `https://github.com/YOUR_USERNAME/CrashSeverityNet.git`)

---

## STEP 3: 아래 명령어 실행 (복사-붙여넣기) (30초)

**PowerShell 또는 터미널**에서 아래 명령어를 **한 줄씩** 실행하세요:

```powershell
# 1) 프로젝트 폴더로 이동
cd "c:/Users/kdksg/Documents/AI TermProject"

# 2) GitHub remote 연결 (URL을 STEP 2에서 복사한 것으로 변경!)
git remote add origin https://github.com/YOUR_USERNAME/CrashSeverityNet.git

# 3) Remote 확인
git remote -v

# 4) Push!
git push -u origin master
```

**GitHub 인증 창이 뜨면**: 
- 브라우저에서 GitHub 로그인
- "Authorize" 클릭

---

## ✅ 완료 확인

Push가 성공하면:
1. `https://github.com/YOUR_USERNAME/CrashSeverityNet` 접속
2. README.md가 예쁘게 렌더링되어 보임
3. 코드 파일들 확인 가능
4. ⚠️ `data/` 폴더 없음 (정상 - .gitignore로 차단됨)

---

## 🔧 문제 해결

### "remote origin already exists" 에러
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/CrashSeverityNet.git
```

### 인증 실패
```bash
# GitHub Personal Access Token 생성 필요
# https://github.com/settings/tokens
# repo 권한 체크 후 생성
# Push 시 비밀번호 대신 Token 입력
```

---

**요약**: GitHub에서 레포 만들고 → URL 복사 → 3줄 명령어 실행 → 끝!
