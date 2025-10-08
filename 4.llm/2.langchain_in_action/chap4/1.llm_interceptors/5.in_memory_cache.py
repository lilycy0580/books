
import time
from langchain_community.llms import Ollama
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

if __name__ == '__main__':
    # 内存缓存
    set_llm_cache(InMemoryCache())
    llm = Ollama(model="qwen:1.8b")

    # 第一次调用
    start_time = time.time()
    output = llm.invoke("中国的首都是哪个城市?")
    end_time = time.time()
    print(f"第一次执行时间: {end_time - start_time:.2f} 秒")
    print(f"输出: {output}")

    print("-" * 50)

    # 第二次调用
    start_time = time.time()
    output = llm.invoke("中国的首都是哪个城市?")
    end_time = time.time()
    print(f"第二次执行时间: {end_time - start_time:.2f} 秒")
    print(f"输出: {output}")
    """
    第一次执行,LLM输出内容没有在缓存中,需要很长事件
    第二次执行,从缓存中调用数据,速度更快
    
    第一次执行时间: 4.84 秒
    输出: 中国的首都是北京。北京市位于中国华北地区，东临渤海，西靠太行山，北濒燕山山脉。总面积1682.54平方千米，其中陆地面积1073.97平方千米，水域面积608.57平方千米。截至2021年底，北京市常住人口达到2102万人。北京是中国的首都，是中华人民共和国中央人民政府所在地、中国政治中心之一，也是世界著名古都和历史文化名城。
    --------------------------------------------------
    第二次执行时间: 0.00 秒
    输出: 中国的首都是北京。北京市位于中国华北地区，东临渤海，西靠太行山，北濒燕山山脉。总面积1682.54平方千米，其中陆地面积1073.97平方千米，水域面积608.57平方千米。截至2021年底，北京市常住人口达到2102万人。北京是中国的首都，是中华人民共和国中央人民政府所在地、中国政治中心之一，也是世界著名古都和历史文化名城。
    """



