# Local Researcher - 기본 사용법

이 문서는 Local Researcher의 기본적인 사용법을 설명합니다.

## 설치 및 설정

### 1. 프로젝트 클론
```bash
git clone <repository-url>
cd local_researcher_project
```

### 2. 환경 설정
```bash
# 자동 설정 스크립트 실행
./scripts/setup.sh

# 또는 수동 설정
npm install
pip install -r requirements.txt
npm install -g @google/gemini-cli
```

### 3. API 키 설정
`.env` 파일을 편집하여 필요한 API 키를 설정합니다:
```bash
# OpenAI API 키
OPENAI_API_KEY=your_openai_api_key_here

# Google API 키 (Gemini용)
GOOGLE_API_KEY=your_google_api_key_here

# 기타 필요한 API 키들...
```

## 기본 사용법

### 1. 간단한 리서치 시작
```bash
# 기본 리서치
gemini research "인공지능의 최신 동향"

# 특정 도메인 리서치
gemini research --domain technology "머신러닝 기술 발전"

# 상세 분석 요청
gemini research --depth comprehensive "기후변화 대응 기술"
```

### 2. 리서치 상태 확인
```bash
# 모든 활성 리서치 확인
gemini status

# 특정 리서치 상태 확인
gemini status research_20240101_1234_5678
```

### 3. 리서치 목록 보기
```bash
# 모든 리서치 보기
gemini list

# 활성 리서치만 보기
gemini list --active

# 완료된 리서치만 보기
gemini list --completed
```

### 4. 리서치 취소
```bash
gemini cancel research_20240101_1234_5678
```

## 고급 사용법

### 1. 대화형 모드
```bash
# 대화형 모드로 시작
gemini research "블록체인 기술" --interactive

# 또는 직접 대화형 모드 진입
gemini
```

### 2. 커스텀 소스 지정
```bash
gemini research "양자컴퓨팅" --sources "arxiv,web,news"
```

### 3. 출력 형식 지정
```bash
# Markdown 형식 (기본)
gemini research "AI 윤리" --format markdown

# PDF 형식
gemini research "AI 윤리" --format pdf

# HTML 형식
gemini research "AI 윤리" --format html
```

### 4. 설정 관리
```bash
# 현재 설정 보기
gemini config --show

# 설정 편집
gemini config --edit

# 설정 초기화
gemini config --reset
```

## 실제 사용 예제

### 예제 1: 기술 동향 리서치
```bash
# AI 기술 동향에 대한 종합적인 리서치
gemini research --domain technology --depth comprehensive "생성형 AI의 현재와 미래"
```

**결과**: 
- 기술 동향 분석
- 주요 플레이어 분석
- 시장 전망
- 기술적 도전과제
- 미래 전망

### 예제 2: 학술 연구 리서치
```bash
# 학술 논문 기반 리서치
gemini research --domain science --depth standard "양자머신러닝의 최신 연구 동향"
```

**결과**:
- 최신 논문 분석
- 연구 방법론
- 주요 발견사항
- 향후 연구 방향

### 예제 3: 비즈니스 리서치
```bash
# 비즈니스 환경 분석
gemini research --domain business --depth comprehensive "전기차 시장의 경쟁 구도 분석"
```

**결과**:
- 시장 규모 및 성장률
- 주요 기업 분석
- 경쟁 구도
- 시장 기회 및 위험 요소
- 투자 전망

## 출력 파일 관리

### 1. 출력 파일 위치
리서치 결과는 다음 위치에 저장됩니다:
```
outputs/
├── research_20240101_1234_5678.md
├── research_20240101_1234_5678.pdf
└── research_20240101_1234_5678.html
```

### 2. 파일 형식별 특징

#### Markdown (.md)
- 가장 기본적인 형식
- 텍스트 에디터에서 편집 가능
- GitHub 등에서 바로 볼 수 있음

#### PDF (.pdf)
- 인쇄 및 공유에 적합
- 포맷팅이 고정됨
- 전문적인 보고서 형태

#### HTML (.html)
- 웹 브라우저에서 볼 수 있음
- 인터랙티브 요소 포함 가능
- 링크 및 이미지 지원

## 문제 해결

### 1. 일반적인 문제들

#### API 키 오류
```bash
Error: API key not found
```
**해결방법**: `.env` 파일에서 API 키를 올바르게 설정했는지 확인

#### 의존성 오류
```bash
Error: Module not found
```
**해결방법**: 의존성을 다시 설치
```bash
npm install
pip install -r requirements.txt
```

#### 권한 오류
```bash
Error: Permission denied
```
**해결방법**: 실행 권한 확인
```bash
chmod +x scripts/setup.sh
```

### 2. 로그 확인
```bash
# 로그 파일 확인
tail -f logs/local_researcher.log

# 상세 로그 레벨 설정
export LOG_LEVEL=DEBUG
```

### 3. 성능 최적화

#### 메모리 사용량 최적화
```bash
# Python 가상환경 사용
source venv/bin/activate

# 불필요한 프로세스 정리
pkill -f "local_researcher"
```

#### 네트워크 최적화
```bash
# 캐시 사용
export CACHE_ENABLED=true

# 병렬 처리 설정
export MAX_WORKERS=4
```

## 팁과 모범 사례

### 1. 효과적인 리서치를 위한 팁

#### 명확한 토픽 정의
```bash
# 좋은 예
gemini research "머신러닝에서의 딥러닝 기술 발전"

# 나쁜 예
gemini research "AI"
```

#### 적절한 깊이 선택
- `basic`: 빠른 개요 (5-10분)
- `standard`: 상세 분석 (15-30분)
- `comprehensive`: 종합 분석 (30-60분)

#### 도메인별 최적화
- `technology`: 기술 동향 및 개발
- `science`: 학술 연구 및 논문
- `business`: 시장 분석 및 비즈니스
- `general`: 일반적인 정보

### 2. 결과 활용

#### 보고서 구조화
```markdown
# 리서치 결과 구조
1. 개요 (Executive Summary)
2. 배경 (Background)
3. 주요 발견사항 (Key Findings)
4. 분석 (Analysis)
5. 결론 (Conclusion)
6. 참고문헌 (References)
```

#### 데이터 시각화
- 차트 및 그래프 추가
- 인포그래픽 생성
- 대시보드 구성

### 3. 협업 및 공유

#### 팀 공유
```bash
# 결과를 팀과 공유
cp outputs/research_*.md /shared/research/

# 실시간 협업
gemini research --collaborative "프로젝트 분석"
```

#### 버전 관리
```bash
# Git을 사용한 버전 관리
git add outputs/
git commit -m "Add research results"
git push
```

## 다음 단계

### 1. 고급 기능 탐색
- 커스텀 워크플로우 생성
- API 통합
- 자동화 스크립트 작성

### 2. 커뮤니티 참여
- GitHub Issues 보고
- 기능 요청 제안
- 코드 기여

### 3. 문서 및 튜토리얼
- `/docs` 디렉토리 확인
- 예제 코드 학습
- 고급 설정 가이드 참조

## 지원 및 도움말

### 1. 도움말 명령어
```bash
# 일반 도움말
gemini help

# 특정 명령어 도움말
gemini research --help
```

### 2. 문서 참조
- `/docs` 디렉토리의 상세 문서
- `/examples` 디렉토리의 예제 코드
- GitHub Wiki의 추가 정보

### 3. 커뮤니티 지원
- GitHub Issues
- Discord 채널
- 이메일 지원

---

이 가이드를 통해 Local Researcher를 효과적으로 사용할 수 있기를 바랍니다. 추가 질문이나 문제가 있으면 언제든지 지원팀에 문의해 주세요! 