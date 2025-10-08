import time

from langchain_community.llms import Ollama
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache

if __name__ == '__main__':
    # SQLite缓存 (需创建db文件夹,.db文件LangChain会自动创建)
    set_llm_cache(SQLiteCache(database_path="./../db/langchain.db"))
    llm = Ollama(model="qwen:1.8b")

    # 第一次调用
    start_time = time.time()
    output = llm.invoke("讲个笑话")
    end_time = time.time()
    print(f"第一次执行时间: {end_time - start_time:.2f} 秒")
    print(f"输出: {output}")

    print("-" * 50)

    # 第二次调用
    start_time = time.time()
    output = llm.invoke("讲个笑话")
    end_time = time.time()
    print(f"第二次执行时间: {end_time - start_time:.2f} 秒")
    print(f"输出: {output}")
    """
    第一次执行时间: 4.47 秒
    输出: 小明家养了一只鹦鹉，每天早上，鹦鹉都会准时在窗户上唱歌。一次，小明听到了鹦鹉的歌声“你是我的最爱”，小明被逗乐了，他开心地笑了起来：“这只鹦鹉真是我的最爱啊！”
    --------------------------------------------------
    第二次执行时间: 0.17 秒
    输出: 小明家养了一只鹦鹉，每天早上，鹦鹉都会准时在窗户上唱歌。一次，小明听到了鹦鹉的歌声“你是我的最爱”，小明被逗乐了，他开心地笑了起来：“这只鹦鹉真是我的最爱啊！”
    """