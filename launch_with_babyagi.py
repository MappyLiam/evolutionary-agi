import os
import faiss
import gradio as gr

from agi.babyagi import BabyAGI
from typing import Optional
from langchain_community.chat_models import ChatOllama
from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore



##### OpenAI API 키 받아오기 -> ChatGPT 엔진
os.environ["OPENAI_API_KEY"] = ""
##### SERP API 키 받아오기 -> 구글 검색 엔진
os.environ["SERPAPI_API_KEY"] = ""


##### embedding model 정의 -> Text embedding 객체 생성
embeddings_model = OpenAIEmbeddings()


##### Initialize the vectorstore
embedding_size = 1536   # Embedding 차원의 수
index = faiss.IndexFlatL2(embedding_size)   # L2 norm 기반의 인덱스
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

##### 둘중에 하나의 모델을 선택
##### LLM을 gemma2:2b로 정의
llm = ChatOllama(model="gemma2:2b", temperature=0)
# temperature은 모델의 답변의 편차를 지정하는 값 -> 0이면 일관된 답변

##### LLM을 OpenAI로 정의
# llm = OpenAI(temperature=0)
# temperature은 모델의 답변의 편차를 지정하는 값 -> 0이면 일관된 답변


##### BabyAGI 설정
# Logging of LLMChains
verbose=False

# If None, will keep on going forever
# Task 진행 반복 횟수
max_iterations: Optional[int] = 3

baby_agi = BabyAGI.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    verbose=verbose,
    max_iterations=max_iterations
)


##### Gradio chatbot -> babyagi 적용한 chatbot
def chat(message, history):
    # invoke 메서드 사용
    response = baby_agi.invoke({"objective": message})
    
    # BabyAGI의 출력은 "task_output" 키에 저장되어 있습니다.
    task_output = response.get("task_output", "No output generated.")
    
    return task_output



#### Gradio 웹 호스팅
demo = gr.ChatInterface(fn=chat, examples=["hello", "hola", "merhaba"], title="Adolescence AGI Bot")
demo.launch(share=True)  # 웹에서 공유 가능하도록 share=True 설정


