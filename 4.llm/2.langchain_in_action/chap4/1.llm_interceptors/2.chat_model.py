from importlib.resources import contents

from langchain.schema import AIMessage,HumanMessage,SystemMessage
from langchain_community.chat_models import ChatOllama

if __name__ == '__main__':
    """
    LangChain支持的消息类型:(主要使用前3种消息类型)
        AIMessage
        HumanMessage
        SystemMessage
        ChatMessage    允许使用任意角色参数   
    """

    chat_model = ChatOllama(model="qwen:1.8b")
    message = HumanMessage(content="把这句话从中文翻译成英文:我喜欢编程")
    response = chat_model([message])
    print(response)
    """
    content='I like programming.' 
    response_metadata={
        'model': 'qwen:1.8b', 
        'created_at': '2025-10-08T09:49:14.945407Z', 
        'message': {'role': 'assistant', 'content': ''}, 
        'done': True, 'done_reason': 'stop', 
        'total_duration': 2617897000, 
        'load_duration': 2429073300, 
        'prompt_eval_count': 18, 
        'prompt_eval_duration': 144487500, 
        'eval_count': 5, 
        'eval_duration': 42138800
    } 
    id='run-2b7053af-41bf-4675-a15c-4c131fdef05a-0'
    """

    messages = [
        SystemMessage(content="你是一个乐于助人的助手,能把中文翻译成英文"),
        HumanMessage(content="我喜欢编程")
    ]
    result = chat_model(messages)
    print(result)
    """
    content='I enjoy programming.' 
    response_metadata={
        'model': 'qwen:1.8b', 
        'created_at': '2025-10-08T09:49:17.2034293Z', 
        'message': {'role': 'assistant', 'content': ''}, 
        'done': True, 
        'done_reason': 'stop', 
        'total_duration': 214493600, 
        'load_duration': 156085200, 
        'prompt_eval_count': 27, 
        'prompt_eval_duration': 22383500, 
        'eval_count': 5, 
        'eval_duration': 34403500
    } 
    id='run-5ee13f4d-67be-4484-9ad2-5f9f0f82691e-0'
    """

    # generate
    batch_messages = [
        [
            SystemMessage(content="你是一个乐于助人的助手,能把中文翻译成英文"),
            HumanMessage(content="我喜欢编程")
        ],
        [
            SystemMessage(content="你是一个乐于助人的助手,能把中文翻译成英文"),
            HumanMessage(content="我爱人工智能")
        ],
    ]
    result = chat_model.generate(batch_messages)
    print(result)
    """
    generations=
        [
            [ChatGeneration(
                text='I enjoy programming.', 
                generation_info={
                    'model': 'qwen:1.8b', 
                    'created_at': '2025-10-08T09:52:10.830631Z', 
                    'message': {'role': 'assistant', 'content': ''}, 
                    'done': True, 
                    'done_reason': 'stop', 
                    'total_duration': 351539700, 
                    'load_duration': 171911800, 
                    'prompt_eval_count': 27, 
                    'prompt_eval_duration': 2010200, 
                    'eval_count': 5, 
                    'eval_duration': 176485400
                }, 
                message=AIMessage(
                    content='I enjoy programming.', 
                    response_metadata={
                        'model': 'qwen:1.8b', 
                        'created_at': '2025-10-08T09:52:10.830631Z', 
                        'message': {'role': 'assistant', 'content': ''}, 
                        'done': True, 
                        'done_reason': 'stop', 
                        'total_duration': 351539700, 
                        'load_duration': 171911800, 
                        'prompt_eval_count': 27, 
                        'prompt_eval_duration': 2010200, 
                        'eval_count': 5, 
                        'eval_duration': 176485400
                    }, 
                    id='run-6143a5a2-af24-419f-a940-130b9622fd9d-0'
                )
            )], 
            [ChatGeneration(text='I love artificial intelligence.', generation_info={'model': 'qwen:1.8b', 'created_at': '2025-10-08T09:52:13.2345756Z', 'message': {'role': 'assistant', 'content': ''}, 'done': True, 'done_reason': 'stop', 'total_duration': 380986800, 'load_duration': 171213900, 'prompt_eval_count': 28, 'prompt_eval_duration': 40348500, 'eval_count': 6, 'eval_duration': 167787800}, message=AIMessage(content='I love artificial intelligence.', response_metadata={'model': 'qwen:1.8b', 'created_at': '2025-10-08T09:52:13.2345756Z', 'message': {'role': 'assistant', 'content': ''}, 'done': True, 'done_reason': 'stop', 'total_duration': 380986800, 'load_duration': 171213900, 'prompt_eval_count': 28, 'prompt_eval_duration': 40348500, 'eval_count': 6, 'eval_duration': 167787800}, id='run-ffba61f9-b447-4060-a9ce-ffa994bfda71-0'))]
        ] 
        llm_output={} 
        run=[RunInfo(run_id=UUID('6143a5a2-af24-419f-a940-130b9622fd9d')), 
             RunInfo(run_id=UUID('ffba61f9-b447-4060-a9ce-ffa994bfda71'))]
    """
