
from langchain_community.llms import Ollama

if __name__ == '__main__':
    # deepseek基础模型(文本补全模型)
    llm = Ollama(
        model="qwen:1.8b",  # 基础语言模型
        base_url="http://localhost:11434",
        temperature=0.1,
        num_predict=1024
    )
    # call()被废弃,使用invoke()替代
    result = llm.invoke("哪些城市曾经是中国的首都")
    print(result)
    """
    中国历史上有多个城市曾作为中国的首都，以下是其中一些重要的城市：
    1. 唐朝长安（今陕西西安）：长安是唐朝的首都，也是世界上最早的、规模宏大的城市之一。长安城内设有众多宫殿、寺庙和官署，是中国古代政治、经济、文化中心。
    2. 北宋东京（今河南开封）：东京是北宋的首都，也是中国历史上最繁华、人口最多的城市之一。东京城内设有众多宫殿、寺庙和官署，是中国古代政治、经济、文化中心。
    3. 明朝南京（今江苏南京）：南京是明朝的首都，也是中国历史上最繁华、人口最多的城市之一。南京城内设有众多宫殿、寺庙和官署，是中国古代政治、经济、文化中心。
    4. 清朝北京（今北京市）：北京是清朝的首都，也是中国历史上最繁华、人口最多的城市之一。北京城内设有众多宫殿、寺庙和官署，是中国古代政治、经济、文化中心。
    5. 19世纪上海（今上海市）：上海是19世纪中国的经济、文化中心，也是中国近代史上重要的海港城市。上海城内设有众多宫殿、寺庙和官署，是中国古代政治、经济、文化中心。
    6. 20世纪北京（今北京市）：北京是20世纪中国的政治、经济、文化中心，也是中国近代史上重要的首都城市。北京城内设有众多宫殿、寺庙和官署，是中国古代政治、经济、文化中心。
    7. 21世纪上海（今上海市）：上海是21世纪中国的经济、文化中心，也是中国近代史上重要的海港城市。上海城内设有众多宫殿、寺庙和官署，是中国古代政治、经济、文化中心。
    8. 20世纪广州（今广州市）：广州是20世纪中国的政治、经济、文化中心，也是中国近代史上重要的海港城市。广州城内设有众多宫殿、寺庙和官署，是中国古代政治、经济、文化中心。
    9. 19世纪福州（今福州市）：福州是19世纪中国的政治、经济、文化中心，也是中国近代史上重要的海港城市。福州城内设有众多宫殿、寺庙和官署，是中国古代政治、经济、文化中心。
    10. 20世纪宁波（今宁波市）：宁波是20世纪中国的政治、经济、文化中心，也是中国近代史上重要的海港城市。宁波城内设有众多宫殿、寺庙和官署，是中国古代政治、经济、文化中心。
    """

    result = llm.generate(["Tell me a joke, Tell me a poem"]*15)
    print(len(result.generations))
    print(result)
    print(result.generations[0])
    print(result.generations[-1])
    print(result.llm_output)    # 获取与模型相关的特定信息  OpenAI有token信息   qwen:1.8b:None
    """
    15
    [GenerationChunk(
        text="Joke:\nWhy don't scientists trust atoms?\nBecause they make up everything!\n
        Poem:\nOde to the Rose\nBy William Shakespeare\nRoses are red, violets are blue,\nAnd everywhere that flowers grow,\nOh, how I love thee, my fair rose,\nThou art not just a flower, but a work of art,\nA masterpiece that will never fade away,\nOh, how I love thee, my fair rose,\nThou art not just a flower, but a work of art,\nA masterpiece that will never fade away,\nOh, how I love thee, my fair rose,\nThou art not just a flower, but a work of art,\nA masterpiece that will never fade away", 
        generation_info={
            'model': 'qwen:1.8b', 'created_at': '2025-10-08T07:51:03.9663152Z', 
            'response': '', 'done': True, 'done_reason': 'stop', 
            'context': [151644, 872, 198, 40451, 752, 264, 21646, 11, 24647, 752, 264, 32794, 151645, 198, 151644, 77091, 198, 41, 4740, 25, 198, 10234, 1513, 6, 83, 13923, 6950, 32199, 30, 198, 17949, 807, 1281, 705, 4297, 0, 198, 32904, 336, 25, 198, 46, 450, 311, 279, 15964, 198, 1359, 12375, 41382, 198, 49, 19696, 525, 2518, 11, 348, 815, 9942, 525, 6303, 11, 198, 3036, 16852, 429, 19281, 3063, 11, 198, 11908, 11, 1246, 358, 2948, 39244, 11, 847, 6624, 16009, 11, 198, 1001, 283, 1947, 537, 1101, 264, 22351, 11, 714, 264, 975, 315, 1947, 11, 198, 32, 58731, 429, 686, 2581, 15016, 3123, 11, 198, 11908, 11, 1246, 358, 2948, 39244, 11, 847, 6624, 16009, 11, 198, 1001, 283, 1947, 537, 1101, 264, 22351, 11, 714, 264, 975, 315, 1947, 11, 198, 32, 58731, 429, 686, 2581, 15016, 3123, 11, 198, 11908, 11, 1246, 358, 2948, 39244, 11, 847, 6624, 16009, 11, 198, 1001, 283, 1947, 537, 1101, 264, 22351, 11, 714, 264, 975, 315, 1947, 11, 198, 32, 58731, 429, 686, 2581, 15016, 3123], 
            'total_duration': 1095391900, 
            'load_duration': 43881200, 
            'prompt_eval_count': 17, 
            'prompt_eval_duration': 5238900, 
            'eval_count': 146, 
            'eval_duration': 1045723600}
        )
    ]
    [GenerationChunk(text="Joke:\nWhy did the tomato turn red?\nBecause it saw the salad dressing!\n
                     Poem:\nOde to a Rose\n\nBy William Shakespeare\n\nRoses are red, violets are blue,\nSuzanne is my favorite, roses are red.\nViolets are blue, and I'm in love with you,\nRoses are red, violets are blue, and I'm in love with you.\nSo here's to you, my dear, roses are red, violets are blue, and I'm in love with you.", 
                     generation_info={'model': 'qwen:1.8b', 'created_at': '2025-10-08T07:51:48.386081Z', 'response': '', 'done': True, 'done_reason': 'stop', 'context': [151644, 872, 198, 40451, 752, 264, 21646, 11, 24647, 752, 264, 32794, 151645, 198, 151644, 77091, 198, 41, 4740, 25, 198, 10234, 1521, 279, 41020, 2484, 2518, 30, 198, 17949, 432, 5485, 279, 32466, 31523, 0, 198, 32904, 336, 25, 198, 46, 450, 311, 264, 15964, 198, 198, 1359, 12375, 41382, 198, 198, 49, 19696, 525, 2518, 11, 348, 815, 9942, 525, 6303, 11, 198, 50, 5197, 20368, 374, 847, 6930, 11, 60641, 525, 2518, 13, 198, 53, 815, 9942, 525, 6303, 11, 323, 358, 6, 76, 304, 2948, 448, 498, 11, 198, 49, 19696, 525, 2518, 11, 348, 815, 9942, 525, 6303, 11, 323, 358, 6, 76, 304, 2948, 448, 498, 13, 198, 4416, 1588, 6, 82, 311, 498, 11, 847, 24253, 11, 60641, 525, 2518, 11, 348, 815, 9942, 525, 6303, 11, 323, 358, 6, 76, 304, 2948, 448, 498, 13], 'total_duration': 844088000, 'load_duration': 46471600, 'prompt_eval_count': 17, 'prompt_eval_duration': 2420200, 'eval_count': 113, 'eval_duration': 795196200})]
    None 
    """
