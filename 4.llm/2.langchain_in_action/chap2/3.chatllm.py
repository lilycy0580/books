
# 导入模板类
from langchain.prompts.chat import (
    ChatPromptTemplate,  # 聊天指令模板的基类
    SystemMessagePromptTemplate,  # 系统消息指令模板的类
    HumanMessagePromptTemplate,  # 人类消息指令模板的类
)
from langchain_community.chat_models import ChatOllama

if __name__ == '__main__':
    # LangChain接入聊天模型,运用预设的指令模板实现翻译应用

    # 1.创建ChatOllama对象,指定LLM
    chat = ChatOllama(model="qwen:1.8b")

    # 2.配置聊天模板信息
    # 设置翻译任务的基本指令
    template = "你是一个翻译助理,请将用户输入的内容由{input_language}直接翻译为{output_language}."
    # 基于template创建系统消息指令,向聊天迷信明确翻译任务要求
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # 简单定义人类消息指令模板
    human_template = "{text}"
    # 定义人类消息指令
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # 将系统消息指令和人类消息指令组合为完整的聊天指令,用于聊天模型交互
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # 3.输入信息并获取翻译结果
    output = chat.invoke(
        chat_prompt.format_prompt(input_language="英语",
                                  output_language="中文",
                                  text="Artificial Intelligence (AI) will have a profound impact on human civilization "
                                       "in the coming years. ").to_messages())
    print(output)
    """
    content='人工智能（AI）将在未来几年对人类文明产生深远影响。' 
    response_metadata={
        'model': 'qwen:1.8b', 
        'created_at': '2025-10-07T09:22:56.6781391Z', 
        'message': {'role': 'assistant', 'content': ''}, 
        'done': True, 
        'done_reason': 'stop', 
        'total_duration': 4256782300, 
        'load_duration': 3980064900, 
        'prompt_eval_count': 50, 
        'prompt_eval_duration': 167768500, 
        'eval_count': 15, 
        'eval_duration': 106022600
    } 
    id='run-854bbf45-3bf5-46b0-be50-4fabb05135f8-0'
    """
