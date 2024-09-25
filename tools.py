import os

from langchain.agents import ZeroShotAgent, Tool
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_community.utilities import SerpAPIWrapper

"""
각각의 템플릿이 프롬프트를 사용자 입력에 맞게 작성되고 이 프롬프트에 대한 LLM 반환값을 순서대로 처리하는 방식

PromptTemplate은 텍스트 생성 모델에게 작업을 요청할 때 사용하는 지침을 미리 작성한 것입니다.
이 템플릿은 미리 정의된 틀과 변수를 사용해 생성될 텍스트의 구조를 정하고, 입력된 데이터를 모델에 전달합니다.
즉, 객체를 입력으로 받는 프롬프트를 미리 설정하여 목표에 맞게 모델의 답변을 유도하는 역할을 합니다.

템플릿 작성 → 입력 값 치환 → LLM 실행 → 결과 반환의 순서
"""


##### OpenAI API 키 받아오기 -> ChatGPT 엔진
os.environ["OPENAI_API_KEY"] = "Insert your OpenAI API Key"
##### SERP API 키 받아오기 -> 구글 검색 엔진
os.environ["SERPAPI_API_KEY"] = "Insert your SERP API Key"


# 프롬프트 템플릿 정의: 주어진 objective에 따른 todo 리스트를 생성하는 템플릿을 생성
todo_prompt = PromptTemplate.from_template("You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}")

# OpenAI 모델을 이용하여 todo_chain을 생성
todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)
# # 만약 Gemma2:2b 모델을 사용하려면 아래의 코드를 사용
# todo_chain = LLMChain(llm = ChatOllama(model="gemma2:2b", temperature=0), prompt=todo_prompt)

# 검색 엔진을 사용하기 위한 SerpAPIWrapper 인스턴스 생성
search = SerpAPIWrapper()

# 사용할 Tool 목록 정의
tools = [
    Tool(
        name = "Search",
        func=search.run,  # SerpAPIWrapper의 run 메서드를 이용하여 검색 실행
        description="useful for when you need to answer questions about current events"  # 이 도구의 사용 목적 설명
    ),
    Tool(
        name = "TODO",
        func=todo_chain.run,  # todo_chain의 run 메서드를 이용하여 todo 리스트 생성
        description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!"
    )
]

# ZeroShotAgent를 위한 프롬프트 템플릿 정의
prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
suffix = """Question: {task}
{agent_scratchpad}"""
prompt = ZeroShotAgent.create_prompt(
    tools,  # 위에서 정의한 도구를 사용
    prefix=prefix,  # 프롬프트 앞부분에 추가할 텍스트
    suffix=suffix,  # 프롬프트 뒷부분에 추가할 텍스트
    input_variables=["objective", "task", "context","agent_scratchpad"]  # 프롬프트에 필요한 변수들
)