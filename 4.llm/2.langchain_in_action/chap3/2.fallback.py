

import httpx
from unittest.mock import patch     # patch用于在测试过程中替换模块中的某些函数
from langchain_openai import ChatOpenAI
from openai import RateLimitError   # 处理速率限制错误
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

if __name__ == '__main__':
    # 底层API出现问题 触发速率限制错误
    request = httpx.Request("GET", "/")
    response = httpx.Response(200, request=request)
    error = RateLimitError("rate limit", response=response, body="")

    # max_retries=0,避免重试
    openai_llm = ChatOpenAI(max_retries=0)      # 主模型
    qwen_llm = ChatOllama(model="qwen:1.8b")    # 备用模型
    llm = qwen_llm.with_fallbacks([qwen_llm]) # 故障转移/降级机制

    # 使用OpenAI的LLM展示错误  略 因为没有api-key
    with patch("openai.resources.chat.completions.Completions.create", side_effect=error):
        try:
            print(openai_llm.invoke("你是谁？"))
        except RateLimitError:
            print("遇到错误")
    """
    遇到错误  出现速率限制错误,直接返回异常
    """

    # 测试回退功能
    with patch("openai.resources.chat.completions.Completions.create", side_effect=error):
        try:
            print(llm.invoke("你是谁？"))
        except RateLimitError:
            print("遇到错误")
    """
    content='我是来自阿里云的大规模语言模型，我叫通义千问。通义千问是阿里巴巴AI研究院自主研发的语言模型，它能够理解和生成人类自然语言，并能进行文本摘要、机器翻译等任务。通义千问还具备丰富的知识库和场景模拟能力，能够为用户提供高质量的信息服务。'
    response_metadata={
        'model': 'qwen:1.8b',
        'created_at': '2025-10-07T11:33:07.108886Z',
        'message': {'role': 'assistant', 'content': ''},
        'done': True,
        'done_reason': 'stop',
        'total_duration': 2747396300,
        'load_duration': 2219456600,
        'prompt_eval_count': 11,
        'prompt_eval_duration': 102491000,
        'eval_count': 68,
        'eval_duration': 424929600
    }
    id='run-fc42135a-e37a-4776-b699-46c91b87cb60-0'
    """

    # 使用具有回退功能的LLM
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你真是一个贴心的助手，总会在回复中附上赞美之词。",
            ),
            ("human", "为什么你喜欢{city}"),
        ]
    )
    chain = prompt | llm
    with patch("openai.resources.chat.completions.Completions.create", side_effect=error):
        try:
            print(chain.invoke({"city": "利川"}))
        except RateLimitError:
            print("Hit error")
    """
    content='我喜欢利川有以下几个原因：\n\n1. 地理位置：利川位于中国西南部，地处巫山、大别山和秦岭三大山脉交汇处，是中国西部重要的交通枢纽之一。\n\n2. 景观丰富：利川地区气候湿润，拥有丰富的自然景观。这里有壮丽的巴东三峡大坝，有神秘的龙宫风景区，还有秀美的木兰山森林公园等。\n\n3. 文化底蕴深厚：利川地区历史悠久，文化底蕴深厚。在这里，你可以欣赏到丰富多彩的非物质文化遗产，如土家族歌舞、土家摆手舞、苗族芦笙舞、侗族大歌、彝族史诗《阿诗玛》等。\n\n4. 市场经济活跃：利川市位于长江上游三峡地区腹地，是重庆市西南部的一座重要的区域中心城市。近年来，利川市在坚持社会主义市场经济体制的前提下，经济社会发展取得了显著成就。如今的利川市，市场经济发展活跃，基础设施建设和生态环境保护工作成效明显。\n\n5. 社会和谐稳定：利川市是一个多民族聚居的地方，社会和谐稳定是其重要特征之一。利川市始终把维护社会稳定作为头等大事来抓，采取了一系列有力措施，有效防范和化解各类重大风险，社会和谐稳定局面日益巩固和发展。' 
    response_metadata={
        'model': 'qwen:1.8b', 
        'created_at': '2025-10-07T11:41:00.9255536Z', 
        'message': {'role': 'assistant', 'content': ''}, 
        'done': True, 
        'done_reason': 'stop', 
        'total_duration': 2128083100, 
        'load_duration': 180542900, 
        'prompt_eval_count': 33, 
        'prompt_eval_duration': 3882000, 
        'eval_count': 273, 
        'eval_duration': 1941509200
    } 
    id='run-4060d2ef-83e8-465e-a7b3-9a298d14ca14-0'
    """