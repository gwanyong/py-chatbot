# Milvus RAG 프로젝트 계획

## 목표
Project Gutenberg의 무료 도서를 사용하여 Milvus 기반 RAG 시스템을 구축하고 정확도를 테스트

## 선택 도서 후보
1. "Alice's Adventures in Wonderland" by Lewis Carroll
2. "The Adventures of Tom Sawyer" by Mark Twain  
3. "Pride and Prejudice" by Jane Austen

## 기술 스택
- **Vector DB**: Milvus
- **Embedding**: OpenAI text-embedding-3-small
- **Framework**: LangChain
- **API**: FastAPI
- **Infrastructure**: Docker

## 구현 단계

### 1. 환경 설정
```bash
# Milvus Docker 실행
docker-compose up -d milvus-standalone

# 패키지 설치
pip install pymilvus langchain openai fastapi uvicorn
```

### 2. 데이터 준비
- Project Gutenberg에서 텍스트 다운로드
- 1000자 단위로 청킹
- 메타데이터 추가 (챕터, 페이지)

### 3. 벡터화 및 인덱싱
- OpenAI API로 임베딩 생성
- Milvus 컬렉션 생성
- 벡터 인덱스 구축

### 4. RAG 시스템 구현
- 질문 임베딩 생성
- 유사도 검색 (top-k)
- 컨텍스트 기반 답변 생성

### 5. 테스트 케이스
- **직접 팩트**: "주인공의 이름은?"
- **추론**: "왜 이런 행동을 했는가?"
- **요약**: "첫 번째 챕터 내용은?"

## 평가 기준
1. **정확성**: 책 내용과 일치하는 답변
2. **완전성**: 충분한 정보 제공
3. **일관성**: 동일 질문에 대한 안정적 답변
4. **응답 시간**: 실용적인 검색 속도

## 예상 결과물
- Milvus 기반 RAG API 서버
- 웹 인터페이스 (선택사항)
- 정확도 테스트 리포트
- 성능 벤치마크 결과