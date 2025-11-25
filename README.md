### SKN19-3rd-1Team

# SK네트웍스 Family AI 캠프 19기 3차 프로젝트

## 1. 공신2 팀 소개

### 팀원
<div align="center">
  <table>
  <tr>
    <td align="center"> 
      <img src="https://avatars.githubusercontent.com/u/117627092?v=4" width="100px;" alt="강지완"/>   
      <br/>
      강지완
      <br/>
      <a href="https://github.com/Maroco0109">
        <img src="https://img.shields.io/badge/GitHub-Maroco0109-181717?style=flat&logo=github&logoColor=white">
      </a>
    </td> 
    <td align="center"> 
      <img src="https://avatars.githubusercontent.com/u/173134983?v=4" width="100px;" alt="김진"/>   
      <br/>
      김진
      <br/>
      <a href="https://github.com/KIMjjjjjjjj">
        <img src="https://img.shields.io/badge/GitHub-KIMjjjjjjjj-181717?style=flat&logo=github&logoColor=white">
      </a>
    </td>
    <td align="center"> 
      <img src="https://avatars.githubusercontent.com/u/109131433?v=4" width="100px;" alt="마한성"/>   
      <br/>
      마한성
      <br/>
      <a href="https://github.com/gitsgetit">
        <img src="https://img.shields.io/badge/GitHub-gitsgetit-181717?style=flat&logo=github&logoColor=white">
      </a>
    </td> 
    <td align="center"> 
      <img src="https://avatars.githubusercontent.com/u/181833818?v=4" width="100px;" alt="Hawon Oh"/>  
      <br/> 
      오하원
      <br/>
      <a href="https://github.com/Hawon-Oh">
        <img src="https://img.shields.io/badge/GitHub-Hawon--Oh-181717?style=flat&logo=github&logoColor=white">
      </a>
    </td> 
  </tr>
</table>
</div>

---

## 2. 프로젝트 개요

### 프로젝트 명
> 공부의 신2 - 대학 학과 탐색 서비스

### 프로젝트 소개
공부의 신2는 선호도/대학 전공/과목 설명을 기반으로 학생에게 적합한 학과와 커리큘럼을 제공하는 AI 서비스입니다.
대학별 전공 과목 데이터를 수집하여 벡터화하고 RAG(Retrieval-Augmented Generation) 구조로 구축했고, LLM의 추론 흐름을 통제하기 위해 LangGraph 기반 ReAct Agent + Structured Pipeline을 결합했습니다. 


### 프로젝트 필요성(배경)
* 정보 파편화
* 융합·신설 학과 증가 
* 정보 비대칭

대부분의 진로 관련 정보는 대학 소개 중심으로 제공되며 학과에서 실제로 무엇을 배우는지 졸업 후 어떤 직무로 이어지는지에 대한 정보는 인터넷에 흩어져 있는 경우가 많아 필요한 정보를 한 번에 찾기 어렵다.  특히 최근 대학들이 다양한 융합·신설 학과를 만들면서 학과명만으로는 실제 커리큘럼이나 진로를 추론하기 점점 어려워지고 있다. 실제로 학생들이 에브리타임 같은 대학교 커뮤니티에 들어가 학과에 대한 질문을 반복적으로 질문하는 경우를 볼 수 있다. 이러한 정보 비대칭으로 인해 많은 학생들이 전공에서 실제로 배우는 내용을 입학 후에야 파악하게 되고 전공 적합성을 뒤늦게 판단하여 쉽게 흥미를 잃는 문제가 발생하고 있다.


### 프로젝트 목표
* 대학별/학과별 정확한 교육과정 데이터 구축
* 학생 관심사 기반 전공/과목 제공
* 학기별 과목 매칭을 통한 커리큘럼 제공
* LangGraph 기반의 Tool-Calling 시스템 구축

---

## 3. 기술 스택 & 사용 모델

### 기술 스택
<p> <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" /> <img src="https://img.shields.io/badge/LangChain-LangGraph-orange?logo=chainlink&logoColor=white" /> <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-4B0082?logo=databricks&logoColor=white" /> <img src="https://img.shields.io/badge/pdfplumber-PDF%20Parsing-lightgrey?logo=adobeacrobatreader&logoColor=white" /> <img src="https://img.shields.io/badge/Selenium-Web%20Automation-43B02A?logo=selenium&logoColor=white" /> </p> 


Frontend

<p> <img src="https://img.shields.io/badge/Streamlit-UI%20Framework-FF4B4B?logo=streamlit&logoColor=white" /> </p> 

Embedding 모델

 <p> <img src="https://img.shields.io/badge/OpenAI-text--embedding--3--small-412991?logo=openai&logoColor=white" /> </p> 

LLM

 <p> <img src="https://img.shields.io/badge/OpenAI-gpt--4o--mini-0A84FF?logo=openai&logoColor=white" /> <img src="https://img.shields.io/badge/Tool%20Calling-Strong%20Support-brightgreen" /> <img src="https://img.shields.io/badge/Korean%20QA-Excellent-blue" /> </p> 

### 깃 구조
```
project name/
│
├── README.md
├── .gitignore
├── .env
├── requirements.txt
│
├── data/                    # 원천데이터 (강의계획서, 학과정보 등)
│   ├── raw/                 # 원본 그대로
│   │   └──seoul_syllabus.json
│   └── processed/           # 전처리 후 데이터
│
├── scripts/                 # 데이터 수집
│   ├── crawling/ 
│   │   └──seoul_major_crawler.py
│   └── preprocessing/
│ 
├── backend/
│   ├── rag/
│   │   ├── retriever/
│   │   ├── embedder/
│   │   └── llm/
│   └── vectorDB/
│
├── frontend/
│
└── docs/                    # 기획 문서
      └── schema.md          # JSON 구조 정의

```

---

## 4. 시스템 아키텍처

<div align="center">
<img width="600" height="1200" alt="image" src="https://github.com/user-attachments/assets/c77d98d8-e563-4898-9d19-4f3d58bbdd1b" />

</div>

본 시스템은 4단계 계층 구조로 설계되어 있으며 확장성과 유연성을 갖춘 AI 아키텍처입니다.

- ✅Frontend: Streamlit 기반의 대화형 사용자 인터페이스
- ✅Backend: LangGraph를 활용한 에이전트 오케스트레이션
- ✅Core Logic: LLM의 추론과 도구(Tool) 실행의 순환 구조
- ✅Data Layer: ChromaDB 기반의 벡터 검색(RAG) 시스템




<div align="center">
<table>
  <tr>
    <td align="center"> 
      <img width="2112" height="945" alt="image" src="https://github.com/user-attachments/assets/39e0844a-1e0e-4e1e-8bf3-c6a1d4aeb39f" />
    </td> 
    <td align="center"> 
      <img width="2112" height="1130" alt="image" src="https://github.com/user-attachments/assets/5c267f69-1104-42f7-966a-52181be39ee1" />
      <br/>
  </tr>
</table>
</div>

---

## 5. WBS (Work Breakdown Structure)

https://www.notion.so/ohgiraffers/WBS-2aa649136c11808d9cb0d22c4d009863

---

## 6. 요구사항 명세서

### 기능적 요구사항
| ID | 요구사항 | 우선순위 | 상태 |
|----|----------|----------|------|
| F-0001 | 사용자 질문 의도 파악 | High | ✅ |
 F-0002 | 사용자의 관심분야 기반 학과 추천 기능 | High | ✅ |
| F-0003 | 학과기반 대학 검색 | High | ✅ |
| F-0004 | 특정 학과/전공 커리큘럼 조회 기능 | High | ✅ |
| F-0005 | 유사학과 커리큘럼 비교 기능 | Medium | 🚧 |
| F-0006 | 학과별 진로 추천 기능 | Low | 📋 |
| F-0007 | 대화 문맥 유지 기능 | High | ✅ |

### 비기능적 요구사항
| ID | 요구사항 | 상태 |
|----|----------|------|
| NF-0001 | 잘못된 정보(할루시네이션) 생성 방지 | 🚧 |
| NF-0002 | 사용자 개인정보 보호 | 📋 |
| NF-0003 | 오류없는 UI 기능구현 | 📋 |
| NF-0004 | 생성할 수 없는 과목설명 존재 시 유저에게 알림 | ✅ |

---

## 7. 수집한 데이터 및 전처리 요약

### 데이터 수집
- **데이터 소스**: 
    - 각 대학 학과별 커리큘럼 사이트
    - 각 대학 강의계획서 사이트
    - 공공데이터포털 https://www.data.go.kr/data/15112124/fileData.do#/tab-layer-file
- **데이터 규모**: 16개 대학의 전공, 수업 데이터 
- **수집 기간**: 2025-11-16 ~ 2025-11-23

### 데이터 전처리
1. **유형1**: 수강신청 페이지 유형별(팝업창/새로운 탭/PDF 뷰어/html) 적절한 방법으로 크롤링
2. **유형2**: PDF 다운로드 후 문서 OCR, 이후 잘못 기입된 newline 및 특수문자 제거
3. **유형3**: 오픈 CSV 데이터셋을 대학/학과/전공별로 JSON 변환, 이후 컬럼명 체크 

### Json 예시
```
{
    "서울대학교": {
        "공과대학": {
            "컴퓨터공학부": [
                {
                    "grade_semester": "3-1",
                    "course_classification": "전필",
                    "name": "알고리즘",
                    "description": "다양한 알고리즘 개발 방법과 알고리즘 분석 기법을 배운다. 귀납적, 재귀적 사고방식을 배우고 이를 통해 문제를 접근하고 해결해나가는 방법을 배운다."
                },
                {
                    "grade_semester": "3-1",
                    "course_classification": "전필",
                    "name": "알고리즘",
                    "description": "다양한 알고리즘 개발 방법과 알고리즘 분석 기법을 배운다. 귀납적, 재귀적 사고방식을 배우고 이를 통해 문제를 접근하고 해결해나가는 방법을 배운다."
                },
            ]
        },
        "자연과학대학": {
            "수리과학부": [
                {
                    "grade_semester": "2-1",
                    "course_classification": "전필",
                    "name": "해석개론 및 연습 1",
                    "description": "완비성 공리를 비롯한 실수체의 기본 성질과 수열의 극한, 상극한과 하극한, 좌표공간의 초보적인 위상적 성질, 코시 수열, 컴팩트 집합과 연결 집합, 함수의 극한과 연속의 엄밀한 정의 및 성질, 고른 연속함수, 단조함수의 성질, 함수열의 고른 수렴, 일변수 함수열의 미분과 적분, 멱급수와 해석함수, 삼각급수, 바이어쉬트라스 점근 정리, 아르젤라-아스콜리 정리, 수열공간 등을 공부한다."
                },
             ]
        }
    }
}
```

---

## 8. DB 연동 구현 코드

<div align="center">
<img width="600" height="267" alt="DB_struct" src="https://github.com/user-attachments/assets/875e9807-d574-43e1-861e-82f957b1ca8a" />
</div>

---

## 9. 테스트 계획 및 결과 보고서

### 테스트 계획
| 테스트 ID | 테스트 항목 | 테스트 방법 | 예상 결과 |
|-----------|------------|------------|----------|
| T-0001 | 환경 및 설치 | 가상환경 설정 | [결과] |
| T-0002 | 데이터 로더 테스트 | JSON 로드 후 반환된 Document의 metadata 필드 확인  | [결과] |
| T-0003 | 임베딩/벡터스토어 테스트 | Chroma DB 생성 | [결과] |
| T-0004 | Retriever 테스트 |  폴백 경로(전공명 필 등) 동작 확인 | [결과] |
| T-0005 |  RAG 파이프라인 테스트 | agent→tools 순환 동작(간단한 tool call) 확인 | [결과] |
| T-0006 | 프론트엔드 테스트 | 질문 입력 → 응답 반환 및 대화 기록 유지 확 | [결과] |
| T-0007 | 성능/비기능 | LLM 응답 시간 측정 | [결과] |

### 테스트 결과
| 테스트 ID | 실행일 | 결과 | 비고 |
|-----------|--------|------|------|
| T-0001 | YYYY-MM-DD | SUCCESS | [비고] |
| T-0002 | YYYY-MM-DD | ✅/❌ | [비고] |

---

## 10. 진행 중인 프로그램 개선 노력

### 개선할 사항 1
- **문제점**: 학과별 진로 추천 기능

- **개선 방안**: 새로운 데이터셋 추가 중 https://www.career.go.kr/cnet/front/openapi/openApiApply04Center.do


### 개선할 사항 2
- **문제점**: 한양대는 컴공이 세부학과로 나눠져있음. 맵핑으로 해결하려 했으나 맵핑에는 한계(비슷한 학과 매핑 시, 'ㄷ'를 먼저 매핑하는 문제가 존재함)
- **개선 방안**: 유사도 기반 학과 검색으로 해결 중

### 개선할 사항 3
- **문제점**: 간헐적으로 LLM에게 학과 정보를 프롬프트로 제공하면 학과 명을 임의로 변경함.
- **개선 방법**: LLM이 복사하기 쉬운 형태로 포매팅 및 System prompt에 학과명 사용 규칙 명시하여 개선했으나 아직 문제점 존재

### 개선할 사항 3
- **문제점**: PDF 문서를 크롤링하는 과정에서 빠진 문장이 존재함. 
- **개선 방법**: 다른 크롤링 방식 시도 중

---

## 11. 수행결과 (테스트/시연 페이지)

### 주요 기능 시연
#### 기능 1. 사용자 관심사 선택
- 대분류: 공학/자연과학
- 소분류: 컴공, 산공

#### 기능 2. 관심사 기반 학과 추천
"내 관심사 기반 학과 추천해줘"
<img width="2880" height="1800" alt="image" src="https://github.com/user-attachments/assets/d40da642-d37e-429e-aa45-458fc50ef639" />

#### 기능 3.  관심 학과 개설 대학 추천
"컴퓨터 관련 학과가 있는 대학을 찾아줘"
- '컴퓨터, 소프트웨어, 인공지능' 중에서 유사도 기반으로 높은 과를 가져옴
<img width="2880" height="1800" alt="image" src="https://github.com/user-attachments/assets/f04b1822-ca88-4971-9ab0-72ad0af546d4" />

#### 기능 4. 관심 학교의 학과 커리큘럼 제공
"서울대학교 컴퓨터공학부 커리큘럼 제공해줘."
- 한 학기마다 최대 다섯개의 과목을 가져와서 커리큘럼을 제공
<img width="2880" height="1800" alt="image" src="https://github.com/user-attachments/assets/ee95524b-658b-43a5-89e4-a5b83f0cbbe9" />

#### 기능 5. 개요 세부 제공
"임베디드시스템과 응용 과목의 자세한 설명을 가져와줘."
- 해당 과목의 개요를 그대로 가져옴
<img width="2880" height="1800" alt="image" src="https://github.com/user-attachments/assets/4f229f91-6ec3-447d-8f73-55f8b6a58d00" />

---

## 12. 한 줄 회고

| 이름 | 회고 |
|------|------|
| 강지완 | [한 줄 회고] |
| 김진 | [한 줄 회고] |
| 마한성 | [한 줄 회고] |
| 오하원 | [한 줄 회고] |

