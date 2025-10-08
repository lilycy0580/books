import json

from langchain_community.llms import Ollama
from langchain.llms.loading import load_llm # 加载预保存的LLM模型的函数,但不支持Ollama模型 deepseek需要充钱,就不演示了

if __name__ == '__main__':
    # LLM参数的报错与加载
    llm = Ollama(model="qwen:1.8b",
                 temperature=0.7,       # 控制随机性 (0-1)
                 top_p=0.9,             # 核采样参数
                 top_k=40,              # 顶部k采样
                 num_predict=512,       # 最大生成长度
                 repeat_penalty=1.1,    # 重复惩罚因子
                 num_ctx=2048           # 上下文窗口大小
    )
    llm._llm_type = "ollama"
    output = llm.invoke("中国的首都")
    print(output)

    llm.save("./../config/llm.json")        # _llm_type:ollama-llm
    llm.save("./../config/llm.yaml")

    llm = load_llm("./../config/llm.json")  # 支持加载的模型(_llm_type:ollama)

