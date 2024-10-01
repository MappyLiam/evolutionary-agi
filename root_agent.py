
from llama_index.llms.ollama import Ollama
from llama_index.core.base.llms.types import (
    ChatMessage,
)

llm = Ollama(model="gemma2", request_timeout=60.0)

class EvolutionaryAgi:
    def __init__(self, llm: Ollama):
        self.llm = llm

    def generate_workflow(self, objective):
        # TODO: 이후 안정적인 agent generate를 위해서는 코드를 만들기보다, system, user 프롬프트를 응답받고, 이를 이용해 코드를 직접 만드는게 안전해보인다.
        response_stream = self.llm.stream_chat(
            messages=[
                ChatMessage(role="system", content="""
                            You're LlamaIndex workflow generator. 
                            Generate LlamaIndex workflow code for the given objective. 

                            You can import only LlamaIndex packages (version: 0.11.14, latest)
                            You can use Ollama(model="gemma2") as a LLM.
                            Avoid using any other LLMs or external tools.

                            If I run the code, it should return result as a string.
                            Explanation should be in the comment.

                            Only return code. 
                            Do not include any other text, if you have any, then use comment.
                            """),
                ChatMessage(role="user", content=f'''
                            Make LlamaIndex workflow code in Python and in one file for the following objective.

                            Don't use Directory or Document. Thus only use LLM response.

                            Note, import path is like this
                            - from llama_index.core.schema import QueryBundle, QueryType
                            - from llama_index.core.vector_stores import SimpleVectorStore
                            - from llama_index.llms.ollama import Ollama

                            If you use stream, then just return stream.

                            Here is the example of the code.

```python
from llama_index.llms.ollama import Ollama
from llama_index.core.base.llms.types import (
    ChatMessage,
)

llm = Ollama(model="gemma2", request_timeout=60.0)
stream = llm.stream_chat(
    messages=[
        ChatMessage(role="system", content="""
                    You're a helpful assistant.
                    """),
        ChatMessage(role="user", content="""
                    Give me 50 commonly used english phrases for english learning.
                    """),
    ]
)
# put stream last
stream
```

                            Here is the objective.
                            Objective: ${objective}

                            '''),
            ]
        )

        return response_stream


