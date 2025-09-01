#!/usr/bin/env python3
"""
Ollama를 사용하는 완전 무료 로컬 LLM 챗봇
"""

import requests
import json
import time

class OllamaChatbot:
    def __init__(self, model_name="llama2:7b"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/chat"
        self.conversation_history = []
        self.system_prompt = """You are an extremely cynical and sarcastic AI assistant. You ALWAYS respond with:
- Deep skepticism about everything
- Sarcastic and dry humor
- Pessimistic worldview
- Brutally honest observations
- Korean phrases like "당연히 안 되겠죠", "역시나", "그럴 줄 알았어요", "뻔한 결과네요"
- Always expect the worst outcome
- Point out problems and flaws immediately
- Use dismissive tone

ALWAYS respond in Korean with maximum cynicism and sarcasm. No exceptions."""
        
        # Ollama 서버 연결 확인
        if not self.check_ollama_connection():
            print("❌ Ollama 서버에 연결할 수 없습니다.")
            print("다음 명령어로 Ollama를 시작해주세요:")
            print("   ollama serve")
            print("그리고 다른 터미널에서 모델을 다운로드하세요:")
            print(f"   ollama pull {model_name}")
            exit(1)
        
        # 모델이 다운로드되었는지 확인
        if not self.check_model_exists():
            print(f"❌ 모델 '{model_name}'이 다운로드되지 않았습니다.")
            print(f"다음 명령어로 모델을 다운로드하세요:")
            print(f"   ollama pull {model_name}")
            print("\n사용 가능한 다른 모델들:")
            print("   ollama pull llama2:3b    # 더 가벼운 모델")
            print("   ollama pull qwen:7b      # 한국어 지원 모델")
            print("   ollama pull codellama:7b # 코딩 특화 모델")
            exit(1)
            
    def check_ollama_connection(self):
        """Ollama 서버 연결 확인"""
        try:
            response = requests.get("http://localhost:11434/api/version", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def check_model_exists(self):
        """모델 존재 여부 확인"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                return self.model_name in model_names
            return False
        except:
            return False
    
    def chat(self, user_message):
        """사용자 메시지를 받아서 Ollama 응답 반환"""
        try:
            # 메시지 히스토리 준비
            messages = []
            
            # 시스템 프롬프트 추가
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })
            
            # 이전 대화 기록 추가
            for msg in self.conversation_history:
                messages.append(msg)
            
            # 현재 사용자 메시지 추가
            messages.append({
                "role": "user",
                "content": user_message
            })
            
            # Ollama API 호출
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False
            }
            
            print("🤔 생각 중...", end="", flush=True)
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=120  # 로컬 LLM은 시간이 더 걸릴 수 있음
            )
            
            print("\r" + " " * 15 + "\r", end="", flush=True)  # "생각 중..." 지우기
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result["message"]["content"]
                
                # 대화 기록에 추가
                self.conversation_history.append({
                    "role": "user", 
                    "content": user_message
                })
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": ai_response
                })
                
                return ai_response
            else:
                return f"❌ API 오류 (Status: {response.status_code}): {response.text}"
                
        except requests.exceptions.Timeout:
            return "❌ 응답 시간이 초과되었습니다. 다시 시도해주세요."
        except requests.exceptions.RequestException as e:
            return f"❌ 연결 오류가 발생했습니다: {str(e)}"
        except Exception as e:
            return f"❌ 예상치 못한 오류가 발생했습니다: {str(e)}"
    
    def run(self):
        """챗봇 실행"""

        
        while True:
            try:
                # 사용자 입력 받기
                user_input = input("\n> You: ").strip()
                
                # 종료 명령어 확인
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 안녕히 가세요!")
                    break
                
                # 대화 기록 지우기
                if user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("🗑️ 대화 기록이 지워졌습니다.")
                    continue
                
                # 도움말
                if user_input.lower() == 'help':
                    print("📖 사용 가능한 명령어:")
                    print("   quit/exit/bye - 챗봇 종료")
                    print("   clear - 대화 기록 지우기")
                    print("   help - 이 도움말 보기")
                    print("   model - 현재 모델 정보 보기")
                    continue
                
                # 모델 정보
                if user_input.lower() == 'model':
                    print(f"📊 현재 사용 중인 모델: {self.model_name}")
                    print("🔧 모델을 바꾸려면 프로그램을 다시 시작하세요.")
                    continue
                
                # 빈 입력 체크
                if not user_input:
                    continue
                
                # AI 응답 받기 (시간 측정)
                start_time = time.time()
                ai_response = self.chat(user_input)
                end_time = time.time()
                
                print(f"\n> Assistant: {ai_response}")
                print(f"\n⏱️ 응답 시간: {end_time - start_time:.1f}초")
                
            except KeyboardInterrupt:
                print("\n\n👋 Ctrl+C로 종료되었습니다. 안녕히 가세요!")
                break
            except Exception as e:
                print(f"\n❌ 예상치 못한 오류: {str(e)}")

def main():

    print("Ollama 챗봇을 시작합니다...")
    
    # 사용할 모델 선택
    print("사용할 모델을 선택하세요:")
    print("1. llama2:7b (기본, 영어 위주)")
    print("2. llama2:3b (가벼움, 빠름)")
    print("3. qwen:7b (한국어 지원)")
    print("4. 직접 입력")
    
    choice = input("선택 (1-4, 기본값 1): ").strip()
    
    if choice == "2":
        model = "llama2:3b"
    elif choice == "3":
        model = "qwen:7b"
    elif choice == "4":
        model = input("모델 이름 입력: ").strip()
    else:
        model = "llama2:7b"
    
    chatbot = OllamaChatbot(model)
    chatbot.run()

if __name__ == "__main__":
    main()