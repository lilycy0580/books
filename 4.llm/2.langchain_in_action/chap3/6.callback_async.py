
import asyncio                                  # 异步编程库,支持非阻塞的异步操作
from typing import Any, Dict, List              # 注解模块,代码的静态类型检查
from langchain_core.messages import HumanMessage# 人类用户输入的消息类  eg:AIMessage,SystemMessage
from langchain_core.outputs import LLMResult    # 封装 LLM 生成结果的类
# from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama  # 调用本地部署的Ollama模型
from langchain.callbacks.base import BaseCallbackHandler, AsyncCallbackHandler  # 同步/异步回调处理器的基类

# 同步回调处理器
class MyCustomSyncHandler(BaseCallbackHandler):
    # 当LLM每生成一个新的token时自动调用   **kwargs:可选参数
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"Sync handler being called in a `thread_pool_executor`: token: {token}")

# 异步回调
class MyCustomAsyncHandler(AsyncCallbackHandler):
    # LLM开始运行时自动调用的异步方法,模型开始生成前执行一些操作
    async def on_llm_start(self,
                           serialized: Dict[str, Any],  # 序列化的LLM配置信息(eg:模型名称,参数等)
                           prompts: List[str],          # 输入的提示词列表(包含所有要发送给模型的文本)
                           **kwargs: Any) -> None:
        print("zzzz....")
        await asyncio.sleep(0.3)        # 异步等待0.3秒
        class_name = serialized["name"] # 从序列化数据获取类名
        print("LLM正在启动:",class_name)

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print("zzzz....")
        await asyncio.sleep(0.3)
        print("LLM结束")

async def main():
    chat = ChatOllama(
        model="qwen:1.8b",  # Ollama 中的模型名称
        base_url="http://localhost:11434",  # Ollama 本地服务地址
        temperature=0,
        streaming=True,
        callbacks=[MyCustomSyncHandler(), MyCustomAsyncHandler()]
    )
    result = await chat.agenerate([[HumanMessage(content="告诉我北京奥运会是在哪一年举办的")]])
    print(result)

if __name__ == '__main__':
    """
    异步API,使用AsyncCallbackHandler处理回调,避免阻塞
    注意:
        同步回调处理程序来处理异步运的链,工具和代理,在后台执行时借助run_in_executor()进行调用
        若同步回调处理程序不是线程安全的,则会引发潜在的运行问题
    """
    asyncio.run(main())
    """
    zzzz....
    LLM正在启动: ChatOllama
    Sync handler being called in a `thread_pool_executor`: token: 2
    Sync handler being called in a `thread_pool_executor`: token: 0
    Sync handler being called in a `thread_pool_executor`: token: 0
    Sync handler being called in a `thread_pool_executor`: token: 8
    Sync handler being called in a `thread_pool_executor`: token: 年
    Sync handler being called in a `thread_pool_executor`: token: 。
    Sync handler being called in a `thread_pool_executor`: token: 
    zzzz....
    LLM结束
    generations=[[ChatGeneration(
        text='2008年。', 
        generation_info={
            'model': 'qwen:1.8b', 
            'created_at': '2025-10-08T04:46:49.9799772Z', 
            'message': {'role': 'assistant', 'content': ''}, 
            'done': True, 
            'done_reason': 'stop', 
            'total_duration': 360659600, 
            'load_duration': 122656300, 
            'prompt_eval_count': 15, 
            'prompt_eval_duration': 1219700, 
            'eval_count': 7, 
            'eval_duration': 235123400
        }, 
        message=AIMessage(
            content='2008年。', 
            response_metadata={'model': 'qwen:1.8b', 
                               'created_at': '2025-10-08T04:46:49.9799772Z', 
                               'message': {'role': 'assistant', 'content': ''}, 
                               'done': True, 
                               'done_reason': 'stop', 
                               'total_duration': 360659600, 
                               'load_duration': 122656300, 
                               'prompt_eval_count': 15, 
                               'prompt_eval_duration': 1219700, 
                               'eval_count': 7, 
                               'eval_duration': 235123400
                               }, 
            id='run-221e5dd9-7f23-45b4-8493-d7d42c86c463-0'
        )
    )]] 
    llm_output={} 
    run=[RunInfo(run_id=UUID('221e5dd9-7f23-45b4-8493-d7d42c86c463'))]
    """



