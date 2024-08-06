# juso-chatbot

## 설치방법

### 1. python 가상 환경 생성, requirements.txt 설치
- 3.11 이상 환경 필수
```py
# 1. generate python bubble
python -m venv env

# 2. activate
source env/bin/activate  # mac
env\Scripts\activate.bat  # window

# 3. install dependencies
pip install -r requirements.txt

# 4. deactivate bubble
# deactivate
```

### 2. `.env` 파일 설정
- `.env.example` 파일 복제 후 `.env` 파일 생성
- 그 외 key 값은 개별 문의

### 3. 데이터베이스 설정
- chroma폴더의 `chroma.sqlite3.zip` 파일 압축 해제
- gitignore 되어 있어서 push해도 깃헙에 업로드 안됨 (100mb 이상 파일이므로)

### 4. 실행
- `chainlit run app.py` 실행 후 `localhost:8000`으로 열리는지 확인
- `.env`파일에서 작성한 아이디와 패스워드로 로그인

### + 추가 (Optional)
- 실행했을 때 로그인 화면이 뜨는 경우
  1. `.env`파일에서 `TESTER_ID`와 `TESTER_PW`에 본인이 로그인할 아이디, 패스워드 입력
  2. Google 로그인 버튼이 있는 경우, Google 계정으로 로그인
- 실행했을 때 로그인 화면이 뜨지 않는다면 해당 과정 생략
