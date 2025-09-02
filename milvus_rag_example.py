"""
Milvus RAG 시스템 예시 코드
내일 구현할 기본 구조
"""

import requests
import openai
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from typing import List, Dict
import re

class MilvusRAG:
    def __init__(self, collection_name: str = "book_chunks"):
        self.collection_name = collection_name
        self.dimension = 1536  # OpenAI text-embedding-3-small dimension
        
    def connect_milvus(self):
        """Milvus 연결"""
        connections.connect("default", host="localhost", port="19530")
        
    def create_collection(self):
        """컬렉션 생성"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="chapter", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="page", dtype=DataType.INT64),
        ]
        
        schema = CollectionSchema(fields, "Book content with embeddings")
        collection = Collection(self.collection_name, schema)
        return collection
        
    def download_gutenberg_book(self, book_id: str) -> str:
        """Project Gutenberg에서 책 다운로드"""
        url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
        response = requests.get(url)
        return response.text
        
    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[Dict]:
        """텍스트를 청크로 분할"""
        # 간단한 문장 단위 분할
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "chapter": "Chapter 1",  # 실제 구현에서는 챕터 파싱
                        "page": len(chunks) + 1
                    })
                current_chunk = sentence + ". "
                
        return chunks
        
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """OpenAI API로 임베딩 생성"""
        embeddings = []
        for text in texts:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embeddings.append(response.data[0].embedding)
        return embeddings
        
    def index_book(self, book_text: str):
        """책 내용을 Milvus에 인덱싱"""
        # 1. 텍스트 청킹
        chunks = self.chunk_text(book_text)
        
        # 2. 임베딩 생성
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.generate_embeddings(texts)
        
        # 3. Milvus에 저장
        collection = Collection(self.collection_name)
        data = [
            [chunk["text"] for chunk in chunks],
            embeddings,
            [chunk["chapter"] for chunk in chunks],
            [chunk["page"] for chunk in chunks]
        ]
        collection.insert(data)
        collection.flush()
        
    def search_similar(self, query: str, top_k: int = 3) -> List[Dict]:
        """유사한 텍스트 검색"""
        # 쿼리 임베딩 생성
        query_embedding = self.generate_embeddings([query])[0]
        
        # Milvus 검색
        collection = Collection(self.collection_name)
        collection.load()
        
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = collection.search(
            [query_embedding], 
            "embedding", 
            search_params, 
            limit=top_k,
            output_fields=["text", "chapter", "page"]
        )
        
        return [
            {
                "text": hit.entity.get("text"),
                "chapter": hit.entity.get("chapter"),
                "page": hit.entity.get("page"),
                "score": hit.score
            }
            for hit in results[0]
        ]
        
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """컨텍스트 기반 답변 생성"""
        context = "\n\n".join([chunk["text"] for chunk in context_chunks])
        
        prompt = f"""Based on the following context from the book, answer the question accurately.

Context:
{context}

Question: {query}

Answer:"""

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        return response.choices[0].message.content
        
    def ask_question(self, query: str) -> Dict:
        """RAG 파이프라인 실행"""
        # 1. 관련 텍스트 검색
        similar_chunks = self.search_similar(query)
        
        # 2. 답변 생성
        answer = self.generate_answer(query, similar_chunks)
        
        return {
            "question": query,
            "answer": answer,
            "sources": similar_chunks
        }

# 사용 예시
if __name__ == "__main__":
    rag = MilvusRAG()
    
    # 앨리스 인 원더랜드 (ID: 11)
    book_text = rag.download_gutenberg_book("11")
    
    # 책 인덱싱
    rag.connect_milvus()
    rag.create_collection()
    rag.index_book(book_text)
    
    # 질문하기
    result = rag.ask_question("Who is the main character?")
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")