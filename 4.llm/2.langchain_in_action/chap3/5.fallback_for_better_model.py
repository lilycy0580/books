
from langchain.output_parsers import DatetimeOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

if __name__ == '__main__':
    prompt =  ChatPromptTemplate.from_template(
        "What time was {event} (in %Y-%m-%dT%H:%M:%S.%f format - only return this value)") # 格式要求与输出限制
    openai_35 = ChatOpenAI() | DatetimeOutputParser()
    openai_4 = ChatOpenAI(model='gpt-4') | DatetimeOutputParser()
    only_35 = prompt | openai_35
    fallback_4 = prompt | openai_35.with_fallbacks(openai_4)
    try:
        only_35.invoke({"event":"the superbowl in 1994"})
    except Exception as e:
        print(e)
    """
    报错
    """

    try:
        fallback_4.invoke({"event": "the superbowl in 1994"})
    except Exception as e:
        print(e)
    """
    正常输出
    """
