
from langchain_openai import ChatOpenAI                 # 导入OpenAI的官方聊天模型
from langchain_community.chat_models import ChatOllama  # 导入Ollama本地聊天模型
from langchain.prompts import ChatPromptTemplate        # 导入聊天提示模板,专门用于构建对话式提示
from langchain_core.prompts import PromptTemplate       # 导入基础提示模板,用于构建单轮提示
from langchain_core.output_parsers import StrOutputParser # 导入字符串输出解析器,处理模型输出

if __name__ == '__main__':
    """
    序列的回退:
        采用两个模型,其中一个模型设置不可用状态(name wrong),另一个模型设置可用状态
        ChatOpenAI 不可用
        本地LLM     可用
    """
    # 处理短输入的模型失效时回退到适用于长处理的模型
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "您是一位贴心的助手，每次回复都会附上赞美之词。",
            ),
            ("human", "为什么你喜欢{city}"),
        ]
    )
    # 使用一个错误模型名称创建一个会报错的链式调用
    # chat_model = ChatOpenAI(model_name="gpt-fake")
    chat_model = ChatOllama(model="gpt-fake")
    bad_chain = chat_prompt | chat_model | StrOutputParser()

    prompt_template = """说明:您应该在回复中始终包含赞美之词.问题:为什么你喜欢{city}?"""
    prompt = PromptTemplate.from_template(prompt_template)
    llm = ChatOllama(model="qwen:1.8b")
    good_chain = prompt | llm

    # 创建一个将两者结合在一起的最终链
    chain = bad_chain.with_fallbacks([good_chain])
    output = chain.invoke({"city": "武汉"})
    print(output)
    """
    content='作为一个生活在城市的人，我深深被武汉的魅力所吸引。以下是几个原因：\n\n1. 丰富的文化历史：武汉是中国历史文化名城之一，拥有丰富的历史文化遗迹和人文景观，如黄鹤楼、东湖公园等，这些都深深地吸引了我。\n\n2. 独特的地理位置：武汉位于长江中游地区，是长江经济带的重要节点城市，具有独特的地理位置优势。这种地理位置优势不仅为武汉的发展提供了坚实的基础，也极大地增强了武汉的魅力，吸引着来自全国各地的人们来到武汉，感受这座城市的魅力和独特之处。' 
    response_metadata={
        'model': 'qwen:1.8b', 
        'created_at': '2025-10-07T13:13:41.3481488Z', 
        'message': {'role': 'assistant', 'content': ''}, 
        'done': True, 
        'done_reason': 'stop', 
        'total_duration': 1253927700, 
        'load_duration': 182284000, 
        'prompt_eval_count': 27, 
        'prompt_eval_duration': 2374900, 
        'eval_count': 119, 
        'eval_duration': 1068487500
    } 
    id='run-680392d6-d839-43a7-a3dd-dae75d8530bc-0'
    """
