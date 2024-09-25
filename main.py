import os
import faiss
import gradio as gr

from babyagi import BabyAGI
from typing import Optional
from langchain_community.chat_models import ChatOllama
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

##### 검색 API key 지정
os.environ["SERPAPI_API_KEY"] = "820dc21a6ac53f30b1807d06aeb07d27aa07d7df0ee9a74cd1f421eef068ef96"


##### embedding model 정의 -> Text embedding 객체 생성
embeddings_model = OpenAIEmbeddings()

# Initialize the vectorstore as empty
embedding_size = 1536   # Embedding 차원의 수
index = faiss.IndexFlatL2(embedding_size)   # L2 norm 기반의 인덱스
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})


##### LLM을 gemma2:2b로 정의
llm = ChatOllama(model="gemma2:2b", temperature=0)
# temperature은 모델의 답변의 편차를 지정하는 값 -> 0이면 일관된 답변


##### BabyAGI 설정
# Logging of LLMChains
verbose=False

# If None, will keep on going forever
# Task 진행 반복 횟수
max_iterations: Optional[int] = 1

baby_agi = BabyAGI.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    verbose=verbose,
    max_iterations=max_iterations
)


# #####Gradio chatbot -> gemma2:2b만을 사용한 chatbot
# def echo(message, history):
    
#     # llm query를 사용자 입력으로 지정
#     # OBJECTIVE = message
    
#     # llm의 답변
#     response = llm.invoke(message)
#     return response.content


##### Gradio chatbot -> babyagi 적용한 chatbot
def echo(message, history):
    
    # run babyagi
    response = baby_agi.invoke({"objective": message})  # invoke를 사용하여 결과 가져오기

    # response를 문자열로 변환하여 반환
    result = response['output'] if 'output' in response else str(response)
    
    return result


#### Gradio 웹 호스팅
demo = gr.ChatInterface(fn=echo, examples=["hello", "hola", "merhaba"], title="Adolescence AGI Bot")
demo.launch(share=True)  # 웹에서 공유 가능하도록 share=True 설정
