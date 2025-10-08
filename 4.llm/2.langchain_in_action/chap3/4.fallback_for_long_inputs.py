
from langchain_openai import ChatOpenAI

if __name__ == '__main__':
    """
    LLM对上下文窗口的Token数量有限制
        LLM发送指令前,可计算跟踪指令的长度 
        复杂情况下,难以准确跟踪指令长度,此时可回退到能够处理更长上下文长度的模型
    """
    inputs = "下一个数字是: " + ", ".join(["one", "two"] * 3000)

    short_llm = ChatOpenAI()
    long_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
    llm = short_llm.with_fallbacks([long_llm])

    try:
        print(short_llm.invoke(inputs))
    except Exception as e:
        print(e)
    """
    报错    
    """

    try:
        print(llm.invoke(inputs))
    except Exception as e:
        print(e)
    """
    正常输出 使用gpt-3.5-turbo-16k模型
    """