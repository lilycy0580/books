import time
import asyncio
from langchain_community.llms import Ollama
from openai import Stream


# 串行 10次
def generate_serially():
    llm = Ollama(model="qwen:1.8b",temperature=0.9) # 温度=0.9 高创造性/随机性
    for _ in range(3):
        resp = llm.generate(['你好,中国首都'])
        print("generate_serially "+str(_)+":"+resp.generations[0][0].text)

# 异步函数,在不阻塞主线程的情况下执行
async def async_generate(llm):
    # await:等待这个异步操作完成,不会阻塞其他任务
    resp = await llm.agenerate(['你好,中国首都'])  # 发送提示"Hello, how are you?"给模型并等待响应
    print("async_generate:"+resp.generations[0][0].text)

# 并发 同时执行多个语言模型调用 (创建10个异步任务并同时启动)
async def generate_concurrently():
    llm = Ollama(model="qwen:1.8b",temperature=0.9)
    # 创建任务列表
    tasks = [async_generate(llm) for _ in range(3)]
    # 并发执行所有任务
    await asyncio.gather(*tasks)

# 主执行部分
async def main_method():
    print("开始并发测试...")
    s = time.perf_counter()
    await generate_concurrently()
    elapsed = time.perf_counter() - s
    print("\033[1m" + f"Concurrent executed in {elapsed:0.2f} seconds." + "\033[0m")

    print("开始串行测试...")
    s = time.perf_counter()
    generate_serially()
    elapsed = time.perf_counter() - s
    print("\033[1m" + f"Serial executed in {elapsed:0.2f} seconds." + "\033[0m")


# 运行主函数
if __name__ == "__main__":
    asyncio.run(main_method())
    """
    开始并发测试...
    async_generate:您好！中国的首都是北京。北京市位于华北地区，是中华人民共和国的首都和最大城市，拥有50多个党政机关、13个中央国家机关驻京机构、24所高校和科研院所。作为中国的首都，北京以其独特的魅力吸引着世界各地的人们前来旅游观光、商务洽谈、学术交流等。北京也是中国历史文化名城之一，有众多的历史文化遗址、古建筑群和名人故居等，如故宫博物院、天安门广场等。总的来说，北京是中国的首都和最大城市，拥有50多个党政机关、13个中央国家机关驻京机构、24所高校和科研院所。同时，北京也是中国历史文化名城之一，有众多的历史文化遗址、古建筑群和名人故居等。这些都说明北京作为中国的首都和最大的城市，具有极高的政治地位、经济实力和社会影响力等方面的优势。
    async_generate:您好！中国首都是指北京市。北京市位于中国的北部，是中华人民共和国的直辖市和国家中心城市。北京是中国历史文化的名城，拥有丰富的文化遗产和美食，如故宫、天坛、长城等。此外，北京还是中国的政治中心和国际大都市，被誉为“东方明珠”。北京以其独特的魅力吸引了全球各地的人士来此居住、工作、学习、旅游等。
    async_generate:你好！中国的首都是北京，位于华北地区。北京是中华人民共和国的首都、中央直辖市，是中国的政治、文化、教育和科研中心。此外，北京还拥有众多世界文化遗产，如故宫、颐和园等。北京作为中国的首都市，是中国的经济、政治、文化和科技中心，有着丰富的历史文化遗产和世界一流的现代化设施。
    Concurrent executed in 2.73 seconds.
    开始串行测试...
    generate_serially 0:您好！中国的首都是北京，位于中国华北地区。北京是中国的首都和政治、经济、文化中心。它拥有丰富的历史文化遗迹，包括故宫、天安门广场、颐和园等。除此之外，北京也是中国最重要的交通枢纽之一。京广线北起北京，南到广州，途经石家庄、郑州、武汉、长沙、湘潭、岳阳等地市。京沪铁路南起上海，北至北京，途经江苏、浙江、安徽等地市。此外，还有像京沈高速这样的高速公路横贯华北地区。总的来说，北京是中国的首都和重要的交通枢纽中心，拥有丰富的历史文化遗迹和独特的地理位置优势。
    generate_serially 1:你好！作为中国的一座城市，北京无疑是中国的首都。下面是一些关于北京的基本信息：1. 地理位置：北京市位于中国的北部，东临渤海，南邻燕山山脉，西邻太行山脉。北京市总面积约2.6万平方千米（占中国陆地面积的50%以上）。2. 面积排名：根据最新的统计数据，北京是中华人民共和国的首都、直辖市和中华人民共和国的边疆城市之一。自1949年新中国成立后，北京市就一直是中国的政治、经济、文化和交通中心，是中华民族的重要象征。 3. 历史文化名城：作为一个拥有5000多年历史的文化名城，北京以其丰富的历史文化资源而闻名于世。北京的名胜古迹众多，包括天安门广场上的天安门城楼、故宫博物院中的景山公园、明长城上的八达岭长城、圆明园内的长春桥和万春亭等。此外，北京还有许多著名的博物馆、艺术馆、图书馆以及各类体育场馆等。总之，作为中国的一座具有5000多年历史的文化名城，北京以其丰富的历史文化资源而闻名于世。
    generate_serially 2:您询问的是中国的首都。中国的首都是北京，位于华北地区，是中华人民共和国的首都和直辖市。   北京位于华北平原北部、太行山脉以西，地势由西北向东南倾斜，总面积16425平方千米（约3.9万平方公里），占全国总面积的1/8左右。北京是中国历史文化名城之一，拥有丰富的历史文化遗产，包括故宫博物院、中国国家博物馆等国家级博物馆和世界文化遗产长城等世界级旅游景点。
    Serial executed in 10.21 seconds.
    """
