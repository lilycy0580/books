
import langchain
from langchain.chains.llm import LLMChain
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

if __name__ == '__main__':
    # 1.不启用debug
    llm = Ollama(model="qwen:1.8b")
    output = llm.invoke("你是谁?")
    print(output)
    """
    我是来自阿里云的超大规模语言模型，我叫通义千问。作为阿里云研发的语言模型，通义千问能够理解和生成多种类型的文字内容，包括但不限于文本描述、故事叙述、观点表达等。
    通义千问的核心能力是基于深度学习和自然语言处理技术构建的全链条对话系统，它具备强大的语言理解能力和精准的口语表达能力，可以满足用户在各种场景下的复杂需求。
    """

    # 2.debug参数调试
    # 所有支持回调的LangChain组件(链,模型,代理,工具,检索器)会输出他们接收的输入和生产的输出
    langchain.debug = True
    # set_debug(True)
    output = llm.invoke("你是谁?")
    """
    [llm/start] [llm:Ollama] Entering LLM run with input:
    {
      "prompts": [
        "你是谁?"
      ]
    }
    [llm/end] [llm:Ollama] [2.99s] Exiting LLM run with output:
    {
      "generations": [
        [
          {
            "text": "我是来自阿里云的大规模语言模型，我叫通义千问。我被设计用来理解和生成人类语言，包括问答、撰写故事、翻译文本等等。\n\n与传统的机器学习模型相比，我的优势在于更大的训练数据集和更先进的自然语言处理技术。这使得我可以更好地理解和生成复杂的语言任务，如回答问题、撰写故事、翻译文本等。\n\n总的来说，我是阿里云自主研发的大型语言模型，能够理解和生成人类语言，具有广泛的应用前景。",
            "generation_info": {
              "model": "qwen:1.8b",
              "created_at": "2025-10-07T11:06:40.6762369Z",
              "response": "",
              "done": true,
              "done_reason": "stop",
              "context": [
                151644,
                ...                     115个
                102653,
                1773
              ],
              "total_duration": 951282600,
              "load_duration": 184034700,
              "prompt_eval_count": 11,
              "prompt_eval_duration": 3988400,
              "eval_count": 101,
              "eval_duration": 761027500
            },
            "type": "Generation"
          }
        ]
      ],
      "llm_output": null,  LLM输出的额外信息,包括总令牌数,指令令牌数,完成令牌数和使用的模型名称
      "run": null
    }
    """

    # 3.verbose参数调试
    # 仅让langchain输出可读性稍高的调试信息,并跳过记录某些原始输出,让用户更专注于应用逻辑
    # 全局范围
    llm = Ollama(model="qwen:1.8b")
    langchain.verbose = True
    # set_verbose(True)
    llm.invoke("你是谁?")

    # 单个对象
    llm = Ollama(model="qwen:1.8b")
    prompt = PromptTemplate(
        input_variables=["question"],
        template="回答以下问题：{question}"
    )
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    output = chain.invoke({"question": "你是谁?"})
    print(output)
    """
    > Entering new LLMChain chain...
    Prompt after formatting:
    回答以下问题：你是谁?
    
    > Finished chain.
    {'question': '你是谁?', 
     'text': '我是来自阿里云的超大规模语言模型，我的名字叫通义千问。通义千问是由阿里巴巴集团研发的大型语言模型，可以进行对话、创作文字和撰写代码等多种任务。通义千问是目前全球最大的语言模型之一，被广泛应用于多个领域，如智能客服、新闻写作、教育咨询等，为人类的生产和生活带来了巨大的便利和推动作用。'}
    """
