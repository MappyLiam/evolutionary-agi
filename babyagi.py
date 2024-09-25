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



    class Config:
        """Configuration for this pydantic object."""
        # Pydantic 일반 Python 객체를 필드로 허용
        arbitrary_types_allowed = True

    # 주어진 task를 task_list에 추가
    def add_task(self, task: Dict):
        self.task_list.append(task)

    # 현재 작업 목록을 출력하고, 문자열로 반환
    def get_task_list(self) -> str:
        # 터미널 출력
        print("\033[95m\033[1m" + "\n***** TASK LIST *****\n" + "\033[0m\033[0m")
        for t in self.task_list:
            print(str(t["task_id"]) + ": " + t["task_name"])
        # ANSI 코드를 제거한 문자열을 반환
        return "\n***** TASK LIST *****\n" + "\n".join([f"{t['task_id']}: {t['task_name']}" for t in self.task_list])

    # 다음으로 수행할 task 출력하고 반환
    def get_next_task(self, task: Dict) -> str:
        # 터미널 출력
        print("\033[92m\033[1m" + "\n***** NEXT TASK *****\n" + "\033[0m\033[0m")
        print(str(task["task_id"]) + ": " + task["task_name"])
        # ANSI 코드를 제거한 문자열을 반환
        return "\n***** NEXT TASK *****\n" + f"{task['task_id']}: {task['task_name']}"

    # 작업 결과를 출력하고 반환
    def get_task_result(self, result: str) -> str:
        # 터미널 출력
        print("\033[93m\033[1m" + "\n***** TASK RESULT *****\n" + "\033[0m\033[0m")
        print(result)
        # ANSI 코드를 제거한 문자열을 반환
        return "\n***** TASK RESULT *****\n" + result





    @property
    # input_keys: BabyAGI 클래스가 기대하는 입력 값(objective)을 지정합니다.
    # output_keys: BabyAGI 클래스가 반환하는 출력 키(task_output)을 지정합니다.
    def input_keys(self) -> List[str]:
        return ["objective"]

    @property
    def output_keys(self) -> List[str]:
        return ["task_output"]  # 출력 키를 'task_output'으로 변경


    # Agent를 활용한 task 수행
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
    # from_llm: BabyAGI 객체를 생성하는 클래스 메서드입니다.
    # ZeroShotAgent: llm_chain과 allowed_tools를 통해 모델을 생성하고 에이전트를 초기화합니다.
    # AgentExecutor: 주어진 agent와 tools를 사용하여 작업을 수행하는 핵심 역할을 담당합니다.
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