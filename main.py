
import json
import re
import gradio as gr

from llama_index.core.schema import QueryBundle, QueryType
from llama_index.llms.ollama import Ollama
from root_agent import EvolutionaryAgi

llm = Ollama(model="gemma2", request_timeout=60.0)

def chat(message, history):
    response_stream = EvolutionaryAgi(llm=llm).generate_workflow(message)
    
    response = ""
    for chunk in response_stream:
        response += chunk.delta
        yield response
    
    code = response
    # to strip ```python and ```
    clean_code = re.sub(r"```[a-zA-Z]*\n?", "", code).strip()
    
    local_scope = {}
    exec(f"""
from llama_index.llms.ollama import Ollama
from llama_index.core.base.llms.types import (
    ChatMessage,
)

def code():
    {clean_code}

result = code()
""", globals(), local_scope)
    stream = local_scope['stream']

    worker_response = response + "\n\nThe code executed successfully.\n\n====Result====\n\n"
    for chunk in stream:
        worker_response += chunk.delta
        yield worker_response


#### Gradio 웹 호스팅
demo = gr.ChatInterface(fn=chat, title="Evolutionary AGI Bot")
demo.launch(share=True)  # 웹에서 공유 가능하도록 share=True 설정


