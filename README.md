# SKN19-3rd-1Team
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
│       └──merged_university_courses.json
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
