from langchain.llms import BaseLLM
from typing import Dict, List
from langchain import LLMChain, PromptTemplate



##### TaskCreationChain 클래스 정의
class TaskCreationChain(LLMChain):
    """Chain to generates tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""

        # 작업 생성에 필요한 템플릿 정의
        # parser로 받아 template을 구성한다. -> objective에 근거하여 agent를 구성
        task_creation_template = (
            "You are an task creation AI that uses the result of an execution agent"
            # 목표 작업을 {objective}에 삽입
            " to create new tasks with the following objective: {objective},"
            # 작업 결과를 {result}에 저장
            " The last completed task has the result: {result}."
            # 작업 결과에 대한 설명
            " This result was based on this task description: {task_description}."
            # 완료되지 않은 작업
            " These are incomplete tasks: {incomplete_tasks}."
            " Based on the result, create new tasks to be completed"
            " by the AI system that do not overlap with incomplete tasks."
            " Return the tasks as an array."
        )
        
        # PromptTemplate을 이용하여 위에서 정의한 템플릿을 구성
        prompt = PromptTemplate(
            template=task_creation_template,
            input_variables=["result", "task_description", "incomplete_tasks", "objective"],
        )
        
        # LLMChain의 인스턴스를 생성하여 반환
        return cls(prompt=prompt, llm=llm, verbose=verbose)
    
    

##### TaskPrioritizationChain 클래스 정의
class TaskPrioritizationChain(LLMChain):
    """Chain to prioritize tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        # 작업 우선순위 결정에 필요한 템플릿 정의
        task_prioritization_template = (
            "You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing"
            " the following tasks: {task_names}."
            " Consider the ultimate objective of your team: {objective}."
            " Do not remove any tasks. Return the result as a numbered list, like:"
            " #. First task"
            " #. Second task"
            " Start the task list with number {next_task_id}."
        )
        
        # PromptTemplate을 이용하여 위에서 정의한 템플릿을 구성
        prompt = PromptTemplate(
            template=task_prioritization_template,
            input_variables=["task_names", "next_task_id", "objective"],
        )
        
        # LLMChain의 인스턴스를 생성하여 반환
        return cls(prompt=prompt, llm=llm, verbose=verbose)


# get_next_task 함수 정의
def get_next_task(task_creation_chain: LLMChain, result: Dict, task_description: str, task_list: List[str], objective: str) -> List[Dict]:
    """다음 작업을 생성하는 함수"""
    
    # 불완전한 작업 목록을 콤마로 구분된 문자열로 만듦
    incomplete_tasks = ", ".join(task_list)
    
    # TaskCreationChain을 통해 새 작업 생성 (주어진 매개변수를 바탕으로 실행)
    response = task_creation_chain.run(result=result, task_description=task_description, incomplete_tasks=incomplete_tasks, objective=objective)
    
    # 응답을 줄바꿈 기준으로 분리하여 리스트로 만듦
    new_tasks = response.split('\n')
    
    # 각 작업을 딕셔너리로 변환하여 반환
    return [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]


# prioritize_tasks 함수 정의
def prioritize_tasks(task_prioritization_chain: LLMChain, this_task_id: int, task_list: List[Dict], objective: str) -> List[Dict]:
    """작업 목록을 우선순위로 정렬하는 함수"""
    
    # 작업 이름을 추출하여 리스트로 만듦
    task_names = [t["task_name"] for t in task_list]
    
    # 다음 작업 ID를 현재 작업 ID보다 하나 증가시킴
    next_task_id = int(this_task_id) + 1
    
    # TaskPrioritizationChain을 통해 작업 목록 우선순위 재정렬 (주어진 매개변수를 바탕으로 실행)
    response = task_prioritization_chain.run(task_names=task_names,
                                             next_task_id=next_task_id,
                                             objective=objective)
    
    # 응답을 줄바꿈 기준으로 분리하여 리스트로 만듦
    new_tasks = response.split('\n')
    
    # 우선순위 작업 목록을 생성
    prioritized_task_list = []
    for task_string in new_tasks:
        if not task_string.strip():
            continue
        
        # 각 작업을 "." 기준으로 분리하여 작업 ID와 작업 이름을 추출
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            prioritized_task_list.append({"task_id": task_id, "task_name": task_name})
    
    # 우선순위가 정렬된 작업 목록을 반환
    return prioritized_task_list


# _get_top_tasks 함수 정의
def _get_top_tasks(vectorstore, query: str, k: int) -> List[str]:
    """주어진 쿼리를 기준으로 상위 k개의 작업을 검색하는 함수"""
    
    # vectorstore에서 쿼리에 대해 k개의 유사한 항목 검색
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    # 결과가 없으면 빈 리스트 반환
    if not results:
        return []
    
    # 검색 결과를 유사도 점수에 따라 내림차순 정렬
    sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))
    
    # 메타데이터의 'task' 항목을 문자열로 추출하여 반환
    return [str(item.metadata['task']) for item in sorted_results]

# execute_task 함수 정의
def execute_task(vectorstore, execution_chain: LLMChain, objective: str, task: str, k: int = 5) -> str:
    """주어진 작업을 실행하는 함수"""
    
    # 주어진 objective를 기준으로 상위 k개의 문맥을 가져옴
    context = _get_top_tasks(vectorstore, query=objective, k=k)
    
    # execution_chain을 실행하여 작업을 수행하고 결과 반환
    return execution_chain.run(objective=objective, context=context, task=task)