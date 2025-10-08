
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler

class MyCustomHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"自定义回调处理器, token: {token}")

if __name__ == '__main__':
    chat = ChatOllama(
        model="qwen:1.8b",                  # 指定模型名称
        base_url="http://localhost:11434",  # Ollama 服务地址
        temperature=0,
        streaming=True,                     # 流式传输模式,逐词(token)返回结果
        callbacks=[MyCustomHandler()]
    )
    # 模型调用方式一:invoke
    result = chat.invoke([HumanMessage(content="中国首都是?")])
    print(result)
    print(result.content)                   # 北京。

    print("----------------------------------------------------------")
    # 模型调用方式二:generate(批量处理)
    result = chat.generate([[HumanMessage(content="中国首都是?")]])
    print(result)
    print(result.generations[0][0].text)    # 北京。

    """
    自定义回调处理器, token: 北京
    自定义回调处理器, token: 。
    自定义回调处理器, token: 
    content='北京。' 
    response_metadata={'model': 'qwen:1.8b', 'created_at': '2025-10-08T05:14:10.2306385Z', 
                       'message': {'role': 'assistant', 'content': ''}, 'done': True, 'done_reason': 'stop', 
                       'total_duration': 229667300, 'load_duration': 41391900, 'prompt_eval_count': 12, 
                       'prompt_eval_duration': 1014900, 'eval_count': 3, 'eval_duration': 186758000} 
    id='run-4a8342cd-19c9-42b4-985c-cc0f414b0c0f-0'
    ----------------------------------------------------------
    自定义回调处理器, token: 北京
    自定义回调处理器, token: 。
    自定义回调处理器, token: 
    generations=[[ChatGeneration(text='北京。', 
                                 generation_info={'model': 'qwen:1.8b', 'created_at': '2025-10-08T05:14:12.4179615Z', 
                                                  'message': {'role': 'assistant', 'content': ''}, 'done': True, 
                                                  'done_reason': 'stop', 'total_duration': 124140100, 
                                                  'load_duration': 49994900, 'prompt_eval_count': 12, 
                                                  'prompt_eval_duration': 1022700, 'eval_count': 3, 
                                                  'eval_duration': 72390600
                                 }, 
                                 message=AIMessage(content='北京。', 
                                                   response_metadata={'model': 'qwen:1.8b', 
                                                                      'created_at': '2025-10-08T05:14:12.4179615Z', 
                                                                      'message': {'role': 'assistant', 'content': ''}, 
                                                                      'done': True, 'done_reason': 'stop', 
                                                                      'total_duration': 124140100, 
                                                                      'load_duration': 49994900, 
                                                                      'prompt_eval_count': 12, 
                                                                      'prompt_eval_duration': 1022700, 
                                                                      'eval_count': 3, 'eval_duration': 72390600
                                                   }, 
                                                   id='run-edc29ff8-e260-4969-875a-adc508ba6993-0'
                                 )
    )]] 
    llm_output={} 
    run=[RunInfo(run_id=UUID('edc29ff8-e260-4969-875a-adc508ba6993'))]
    """