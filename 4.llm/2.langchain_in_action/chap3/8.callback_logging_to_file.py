
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_community.llms import Ollama             # 基础模型
from langchain_community.chat_models import ChatOllama  # 对话模型
from langchain.chains import LLMChain
from langchain.callbacks import FileCallbackHandler
from loguru import logger

# 这个链将同时向标准输出打印（因为 verbose=True）并写入 'output.log'
# 如果 verbose=False，FileCallbackHandler 仍会写入 'output.log'


if __name__ == '__main__':
    """
    FileCallbackHandler   控制日志输出(将日志信息写入文件)
    StdOutCallbackHandler 将日志信息输出到终端
    loguru 日志库,记录处理过程中未被捕获的其他输出信息   pip install loguru
    """

    # 配置logger:添加一个日志处理器
    logfile = "./log/output.log"
    logger.add(logfile, colorize=False, enqueue=True, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")    # 显示彩色日志,线程安全

    # 创建LangChain回调处理器(将LLM的执行过程记录到日志中)
    handler = FileCallbackHandler(logfile)

    llm = Ollama(
        model="qwen:1.8b",  # 指定 Ollama 模型名称
        base_url="http://localhost:11434",  # Ollama 服务地址
        temperature=0
    )
    # llm = ChatOllama(
    #     model="qwen:1.8b",
    #     base_url="http://localhost:11434",
    #     temperature=0
    # )

    # 提示词模板
    prompt = PromptTemplate.from_template("1 + {number} = ")
    # 构建链
    chain = LLMChain(llm=llm, prompt=prompt,
                     callbacks=[handler],    # FileCallbackHandler会记录到文件
                     verbose=False)          # True:控制台输出llm执行过程   False:不输出llm执行过程 仅有info信息
    # 运行链
    answer = chain.run(number=1)
    # 记录结果到日志
    logger.info(answer)
    """ 
    > Entering new LLMChain chain...              ------------>
    Prompt after formatting:
    1 + 1 =                                                         此部分是llm执行过程
    
    > Finished chain.                             ------------>
    2025-10-08 13:33:36.331 | INFO     | __main__:<module>:38 - 2   logger.info(answer)
    """