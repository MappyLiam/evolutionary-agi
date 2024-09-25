from collections import deque
from pydantic import BaseModel, Field
from langchain.llms import BaseLLM
from langchain import LLMChain
from langchain.chains.base import Chain
from langchain.agents import ZeroShotAgent, AgentExecutor
from typing import Dict, List, Optional, Any
from langchain.vectorstores.base import VectorStore

# tools와 prompt 변수 import
from tools import tools, prompt

# BabyAGI에 필요한 class import
from task import TaskCreationChain, TaskPrioritizationChain

# BabyAGI에 필요한 def import
from task import execute_task, get_next_task, prioritize_tasks











##### BabyAGI 에이전트 모델 클래스
class BabyAGI(Chain, BaseModel):
    
    # Chain과 BaseModel을 상속
    # Objective를 받아서 수행
    """Controller model for the BabyAGI agent."""

    # deque를 사용하여 관리되는 작업 목록
    # Field 함수는 데이터 타입을 쉽게 저장해주는 도구 -> 강력한 타입 검사, 유효성 검사
    task_list: deque = Field(default_factory=deque)
    # 작업을 생성하는 chain
    task_creation_chain: TaskCreationChain = Field(...)
    # 작업의 우선순위를 재설정하는 chain
    task_prioritization_chain: TaskPrioritizationChain = Field(...)
    # 작업을 수행하는 데 사용되는 에이전트 실행자
    execution_chain: AgentExecutor = Field(...)
    # 현재 작업 ID counter
    task_id_counter: int = Field(1)
    # 작업 결과 저장 -> vector store
    vectorstore: VectorStore = Field(init=False)
    # 최대 반복 횟수
    max_iterations: Optional[int] = None




    # pydantic 라이브러리 모델 설정
    class Config:
        """Configuration for this pydantic object."""
        # 임의의 데이터 타입 사용 가능
        arbitrary_types_allowed = True

    def add_task(self, task: Dict):
        self.task_list.append(task)

    def print_task_list(self):
        print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
        for t in self.task_list:
            print(str(t["task_id"]) + ": " + t["task_name"])

    def print_next_task(self, task: Dict):
        print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
        print(str(task["task_id"]) + ": " + task["task_name"])

    def print_task_result(self, result: str):
        print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
        print(result)

    @property
    def input_keys(self) -> List[str]:
        return ["objective"]

    @property
    def output_keys(self) -> List[str]:
        return []

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent."""
        objective = inputs['objective']
        first_task = inputs.get("first_task", "Make a todo list")
        self.add_task({"task_id": 1, "task_name": first_task})
        num_iters = 0

        # 특정 결과를 얻을 때까지 while문을 실행
        # OpenAI 등 여러가지 LLM 모델과 같이 사용 가능
        while True:
            if self.task_list:
                self.print_task_list()

                # Step 1: Pull the first task
                task = self.task_list.popleft()
                self.print_next_task(task)

                # Step 2: Execute the task
                result = execute_task(
                    self.vectorstore, self.execution_chain, objective, task["task_name"]
                )
                this_task_id = int(task["task_id"])

                self.print_task_result(result)

                # Step 3: Store the result in Pinecone
                # Pinecone이나 DB를 이용하지 않고 로컬에 저장
                result_id = f"result_{task['task_id']}"
                self.vectorstore.add_texts(
                    texts=[result],
                    metadatas=[{"task": task["task_name"]}],
                    ids=[result_id],
                )

                # Step 4: Create new tasks and reprioritize task list
                new_tasks = get_next_task(
                    self.task_creation_chain, result, task["task_name"], [t["task_name"] for t in self.task_list], objective
                )
                for new_task in new_tasks:
                    self.task_id_counter += 1
                    new_task.update({"task_id": self.task_id_counter})
                    self.add_task(new_task)
                self.task_list = deque(
                    prioritize_tasks(
                        self.task_prioritization_chain, this_task_id, list(self.task_list), objective
                    )
                )
            num_iters += 1
            if self.max_iterations is not None and num_iters == self.max_iterations:
                print("\033[91m\033[1m" + "\n*****TASK ENDING*****\n" + "\033[0m\033[0m")
                break
        return {}

    @classmethod
    def from_llm(
        cls,
        llm: BaseLLM,
        vectorstore: VectorStore,
        verbose: bool = False,
        **kwargs
    ) -> "BabyAGI":
        """Initialize the BabyAGI Controller."""
        task_creation_chain = TaskCreationChain.from_llm(
            llm, verbose=verbose
        )
        task_prioritization_chain = TaskPrioritizationChain.from_llm(
            llm, verbose=verbose
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        tool_names = [tool.name for tool in tools]
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
        return cls(
            task_creation_chain=task_creation_chain,
            task_prioritization_chain=task_prioritization_chain,
            execution_chain=agent_executor,
            vectorstore=vectorstore,
            **kwargs
        )





##### BabyAGI 에이전트 모델 클래스
class BabyAGI(Chain, BaseModel):
    
    # Chain과 BaseModel을 상속
    # Objective를 받아서 수행
    """Controller model for the BabyAGI agent."""

    # deque를 사용하여 관리되는 작업 목록
    # Field 함수는 데이터 타입을 쉽게 저장해주는 도구 -> 강력한 타입 검사, 유효성 검사
    task_list: deque = Field(default_factory=deque)
    task_creation_chain: TaskCreationChain = Field(...)
    task_prioritization_chain: TaskPrioritizationChain = Field(...)
    execution_chain: AgentExecutor = Field(...)
    task_id_counter: int = Field(1)
    vectorstore: VectorStore = Field(init=False)
    max_iterations: Optional[int] = None

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def add_task(self, task: Dict):
        self.task_list.append(task)

    def get_task_list(self) -> str:
        # 터미널 출력
        print("\033[95m\033[1m" + "\n***** TASK LIST *****\n" + "\033[0m\033[0m")
        for t in self.task_list:
            print(str(t["task_id"]) + ": " + t["task_name"])
        # ANSI 코드를 제거한 문자열을 반환
        return "\n***** TASK LIST *****\n" + "\n".join([f"{t['task_id']}: {t['task_name']}" for t in self.task_list])

    def get_next_task(self, task: Dict) -> str:
        # 터미널 출력
        print("\033[92m\033[1m" + "\n***** NEXT TASK *****\n" + "\033[0m\033[0m")
        print(str(task["task_id"]) + ": " + task["task_name"])
        # ANSI 코드를 제거한 문자열을 반환
        return "\n***** NEXT TASK *****\n" + f"{task['task_id']}: {task['task_name']}"

    def get_task_result(self, result: str) -> str:
        # 터미널 출력
        print("\033[93m\033[1m" + "\n***** TASK RESULT *****\n" + "\033[0m\033[0m")
        print(result)
        # ANSI 코드를 제거한 문자열을 반환
        return "\n***** TASK RESULT *****\n" + result

    @property
    def input_keys(self) -> List[str]:
        return ["objective"]

    @property
    def output_keys(self) -> List[str]:
        return ["task_output"]  # 출력 키를 'task_output'으로 변경

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent."""
        objective = inputs['objective']
        first_task = inputs.get("first_task", "Make a todo list")
        self.add_task({"task_id": 1, "task_name": first_task})
        num_iters = 0

        task_output = ""  # 전체 출력을 저장할 변수

        # 특정 결과를 얻을 때까지 while문을 실행
        while True:
            if self.task_list:
                # Step 1: Pull the first task and get task list
                task_list_str = self.get_task_list()
                task_output += task_list_str + "\n"

                # Step 2: Pull the next task
                task = self.task_list.popleft()
                next_task_str = self.get_next_task(task)
                task_output += next_task_str + "\n"

                # Step 3: Execute the task
                result = execute_task(
                    self.vectorstore, self.execution_chain, objective, task["task_name"]
                )
                this_task_id = int(task["task_id"])

                # Step 4: Get task result
                task_result_str = self.get_task_result(result)
                task_output += task_result_str + "\n"

                # Step 5: Store the result in Pinecone
                result_id = f"result_{task['task_id']}"
                self.vectorstore.add_texts(
                    texts=[result],
                    metadatas=[{"task": task["task_name"]}],
                    ids=[result_id],
                )

                # Step 6: Create new tasks and reprioritize task list
                new_tasks = get_next_task(
                    self.task_creation_chain, result, task["task_name"], [t["task_name"] for t in self.task_list], objective
                )
                for new_task in new_tasks:
                    self.task_id_counter += 1
                    new_task.update({"task_id": self.task_id_counter})
                    self.add_task(new_task)

                self.task_list = deque(
                    prioritize_tasks(
                        self.task_prioritization_chain, this_task_id, list(self.task_list), objective
                    )
                )
                
            num_iters += 1
            if self.max_iterations is not None and num_iters == self.max_iterations:
                print("\033[91m\033[1m" + "\n***** TASK ENDING *****\n" + "\033[0m\033[0m")
                task_output += "\n***** TASK ENDING *****\n"
                break

        return {"task_output": task_output}  # 최종 출력 반환
    
    

    
    

    @classmethod
    def from_llm(
        cls,
        llm: BaseLLM,
        vectorstore: VectorStore,
        verbose: bool = False,
        **kwargs
    ) -> "BabyAGI":
        """Initialize the BabyAGI Controller."""
        task_creation_chain = TaskCreationChain.from_llm(
            llm, verbose=verbose
        )
        task_prioritization_chain = TaskPrioritizationChain.from_llm(
            llm, verbose=verbose
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        tool_names = [tool.name for tool in tools]
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
        return cls(
            task_creation_chain=task_creation_chain,
            task_prioritization_chain=task_prioritization_chain,
            execution_chain=agent_executor,
            vectorstore=vectorstore,
            **kwargs
        )