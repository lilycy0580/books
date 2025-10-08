import warnings
from typing import Any, Dict, List, Union
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.agents import AgentAction
from langchain_community.llms import Ollama

# 自定义回调处理器
class MyCustomHandlerOne(BaseCallbackHandler):
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:   # Any:任意类型
        print(f"handler1: on_llm_start {serialized['name']}")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        print(f"handler1: on_new_token {token}")

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Run when LLM errors."""

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        print(f"handler1: on_chain_start {serialized['name']}")

    # Tool开始事件,工具开始时触发,用于跟踪工具的使用
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        print(f"handler1: on_tool_start {serialized['name']}")

    # Agent动作事件,智能体执行动作时触发
    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        print(f"handler1: on_agent_action {action}")

class MyCustomHandlerTwo(BaseCallbackHandler):
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        print(f"handler2: on_llm_start (I'm the second handler!!) {serialized['name']}")

if __name__ == '__main__':

    warnings.filterwarnings("ignore")
    # 实例化处理器
    handler1 = MyCustomHandlerOne()
    handler2 = MyCustomHandlerTwo()

    # 创建AI智能体
    # a:初始化模型
    llm = Ollama(model="qwen:1.8b", callbacks=[handler2])
    # b:加载工具,加载预定义的工具集
    tools = load_tools(["llm-math"], llm=llm)
    # c:创建智能体
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)   # agent:智能体类型 零样本ReAct模式
    # d:使用智能体
    agent.run("为什么影子之间会有吸引力?", callbacks=[handler1])
    """
    只有llm相关的事件触发handler2   包括llm直接调用,工具中llm调用           即handler2对llm对象的所有调用都有效
    整个智能体执行过程会触发handler1 包括chain事件,tool事件,agent事件调用
    
    handler1: on_chain_start AgentExecutor
    handler1: on_chain_start LLMChain
    handler1: on_llm_start Ollama
    handler2: on_llm_start (I'm the second handler!!) Ollama
    handler1: on_new_token  我
    handler1: on_new_token 会
    handler1: on_new_token 回答
    handler1: on_new_token 。
    Final
    handler1: on_new_token  Answer
    handler1: on_new_token :
    handler1: on_new_token  影
    handler1: on_new_token 子
    handler1: on_new_token 之间
    ...
    handler1: on_new_token 会有
    handler1: on_new_token 吸引力
    handler1: on_new_token 。 
    """