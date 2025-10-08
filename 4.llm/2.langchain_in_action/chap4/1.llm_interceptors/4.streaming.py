

from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage,SystemMessage

if __name__ == '__main__':
    # 基础模型
    llm = Ollama(
        model="qwen:1.8b",
        callbacks=[StreamingStdOutCallbackHandler()]    # 注意:不能设置streaming参数
    )
    # 流式输出
    response = llm("请使用跳舞这个词汇造个句子")
    print(response)
    print("-----------------------------------------------------------------------------------------------------\n")

    response = llm.invoke("请使用跳舞这个词汇造个句子")
    print(response)
    print("-----------------------------------------------------------------------------------------------------\n")

    response = llm.generate(["请使用跳舞这个词汇造个句子"])
    print(response)
    print("-----------------------------------------------------------------------------------------------------\n")

    # 聊天模型
    messages = [HumanMessage(content="请使用跳舞这个词汇造个句子")]
    chat = ChatOllama(
        model="qwen:1.8b",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )

    response = chat.invoke(messages)
    print(response)
    print("-----------------------------------------------------------------------------------------------------\n")

    """
    他跳起一支欢快的舞蹈，把整个客厅装点得如同仙境。他跳起一支欢快的舞蹈，把整个客厅装点得如同仙境。
    -----------------------------------------------------------------------------------------------------
    她天生就是一位专业的舞者，她的每一次转身、跳跃都如同舞蹈家的表演一般精彩绝伦。她天生就是一位专业的舞者，她的每一次转身、跳跃都如同舞蹈家的表演一般精彩绝伦。
    -----------------------------------------------------------------------------------------------------
    他跳着欢快的舞步，满脸的笑容。generations=[[GenerationChunk(text='他跳着欢快的舞步，满脸的笑容。', generation_info={'model': 'qwen:1.8b', 'created_at': '2025-10-08T10:45:56.5943746Z', 'response': '', 'done': True, 'done_reason': 'stop', 'context': [151644, 872, 198, 14880, 37029, 112084, 99487, 110376, 66078, 18947, 109949, 151645, 198, 151644, 77091, 198, 42411, 100421, 99164, 115506, 9370, 100066, 64682, 3837, 112377, 110247, 1773], 'total_duration': 492874400, 'load_duration': 181688500, 'prompt_eval_count': 16, 'prompt_eval_duration': 3942200, 'eval_count': 12, 'eval_duration': 306105900})]] llm_output=None run=[RunInfo(run_id=UUID('3c973c3a-350e-4fce-8136-e7e20897867e'))]
    -----------------------------------------------------------------------------------------------------
    他擅长跳舞，他的舞步非常优美，让人看得如痴如醉。
    content='他擅长跳舞，他的舞步非常优美，让人看得如痴如醉。' 
    response_metadata={'model': 'qwen:1.8b', 'created_at': '2025-10-08T10:45:59.0523597Z', 'message': {'role': 'assistant', 'content': ''}, 'done': True, 'done_reason': 'stop', 'total_duration': 438112500, 'load_duration': 181132900, 'prompt_eval_count': 16, 'prompt_eval_duration': 1068000, 'eval_count': 18, 'eval_duration': 254762200} id='run-d63cc3eb-e5bf-4c85-9b51-490c06a5413d-0'
    -----------------------------------------------------------------------------------------------------
    """


