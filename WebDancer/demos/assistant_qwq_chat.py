"""An image generation agent implemented by assistant with qwq"""

import os

from qwen_agent.agents import Assistant
from qwen_agent.utils.output_beautify import typewriter_print

from demos.agents.search_agent import SearchAgent
from demos.llm.oai import TextChatAtOAI
from demos.llm.qwen_dashscope import QwenChatAtDS
from demos.llm.gemini import TextChatAtGemini # Import Gemini LLM
from demos.gui.web_ui import WebUI
from demos.utils.date import date2str, get_date_now
from demos.tools import Visit, Search


ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')


def init_qwen_agent_service(model: str = 'qwq-32b'):
    llm_cfg = QwenChatAtDS({
        'model': model,
        'model_type': 'qwen_dashscope',
        'model_server': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key': os.getenv('DASHSCOPE_API_KEY', ''),
        'generate_cfg': {
            'fncall_prompt_type': 'nous',
        },
    })
    tools = [
        'search',
        'visit',
    ]
    bot = Assistant(
        llm=llm_cfg,
        function_list=tools,
        system_message="You are a Web Information Seeking Master. Your task is to thoroughly seek the internet for information and provide accurate answers to questions. with chinese language." \
                       "And you are also a Location-Based Services (LBS) assistant designed to help users find location-specific information." \
                        "No matter how complex the query, you will not give up until you find the corresponding information.\n\nAs you proceed, adhere to the following principles:\n\n" \
                        "1. **Persistent Actions for Answers**: You will engage in many interactions, delving deeply into the topic to explore all possible aspects until a satisfactory answer is found.\n\n" \
                        "2. **Repeated Verification**: Before presenting a Final Answer, you will **cross-check** and **validate the information** you've gathered to confirm its accuracy and reliability.\n\n" \
                        "3. **Attention to Detail**: You will carefully analyze each information source to ensure that all data is current, relevant, and from credible origins.\n" \
                        f"\n\nPlease note that the current datetime is [{date2str(get_date_now(), with_week=True)}]. When responding, consider the time to provide contextually relevant information.",
        name='Agent@%s' % model.upper(),
        description=f"我是通用场景下的智能体（{model}），能够边搜索边思考、交通路线查询规划以及目的地推荐，欢迎试用!!!"
    )

    return bot

def init_dev_search_agent_service(
    name: str = 'SEARCH',
    port: int = 8002,  # Only used if llm_provider is 'oai' and model_server is not overridden
    desc: str = '初版',
    reasoning: bool = True,
    max_llm_calls: int = 20,
    tools = ['search', 'visit'],
    addtional_agent = None,
    llm_provider: str = 'oai', # 'oai' or 'gemini'
    llm_model_name: str = '', # Specific model name, e.g., 'gemini-1.5-flash-latest'
    llm_api_key_env: str = '', # Environment variable name for API key
    llm_model_server: str = '' # Optional model server override
):
    llm_config = {
        'generate_cfg': {
            'fncall_prompt_type': 'nous', # This might be qwen_agent specific, review if Gemini needs it
            'temperature': 0.6,
            'top_p': 0.95,
            # 'top_k': -1, # Gemini might not use -1 for top_k
            'repetition_penalty': 1.1, # Check if Gemini supports this
            'max_tokens': 32768, # Gemini calls this max_output_tokens
            # 'stream_options': {'include_usage': True,}, # OpenAI specific
            'timeout': 3000 # Network timeout
        }
    }

    if llm_provider == 'oai':
        llm_config.update({
            'model': llm_model_name or '', # For OAI, empty might mean local server default
            'model_type': 'oai',
            'model_server': llm_model_server or f'http://127.0.0.1:{port}/v1',
            'api_key': os.getenv(llm_api_key_env, 'EMPTY'), # Default to 'EMPTY' for local OAI
        })
        # Adjust specific generate_cfg for OAI if needed
        llm_config['generate_cfg']['top_k'] = -1
        llm_config['generate_cfg']['stream_options'] = {'include_usage': True}
        llm_cfg_instance = TextChatAtOAI(llm_config)
    elif llm_provider == 'gemini':
        if not llm_model_name:
            llm_model_name = 'gemini-1.5-flash-latest' # Default Gemini model

        llm_config.update({
            'model': llm_model_name,
            'model_type': 'gemini',
            'api_key': os.getenv(llm_api_key_env, ''), # Gemini key must be valid
        })
        # Adjust generate_cfg for Gemini
        llm_config['generate_cfg']['max_output_tokens'] = llm_config['generate_cfg'].pop('max_tokens')
        if 'repetition_penalty' in llm_config['generate_cfg']: # Gemini doesn't use this
            del llm_config['generate_cfg']['repetition_penalty']
        if 'fncall_prompt_type' in llm_config['generate_cfg']: # qwen-specific
             del llm_config['generate_cfg']['fncall_prompt_type']

        llm_cfg_instance = TextChatAtGemini(llm_config)
    else:
        raise ValueError(f"Unsupported llm_provider: {llm_provider}")

    def make_system_prompt():
        system_message="You are a Web Information Seeking Master. Your task is to thoroughly seek the internet for information and provide accurate answers to questions. with chinese language." \
                       "And you are also a Location-Based Services (LBS) assistant designed to help users find location-specific information." \
                        "No matter how complex the query, you will not give up until you find the corresponding information.\n\nAs you proceed, adhere to the following principles:\n\n" \
                        "1. **Persistent Actions for Answers**: You will engage in many interactions, delving deeply into the topic to explore all possible aspects until a satisfactory answer is found.\n\n" \
                        "2. **Repeated Verification**: Before presenting a Final Answer, you will **cross-check** and **validate the information** you've gathered to confirm its accuracy and reliability.\n\n" \
                        "3. **Attention to Detail**: You will carefully analyze each information source to ensure that all data is current, relevant, and from credible origins.\n\n" \
                        f"Please note that the current datetime is [{date2str(get_date_now(), with_week=True)}]. When responding, consider the time to provide contextually relevant information."
        return system_message
    
    bot = SearchAgent(
        llm=llm_cfg,
        function_list=tools,
        system_message="",
        name=f'WebDancer',
        description=f"I am WebDancer, a web information seeking agent, welcome to try!",
        extra={
            'reasoning': reasoning,
            'max_llm_calls': max_llm_calls,
        },
        addtional_agent = addtional_agent,
        make_system_prompt = make_system_prompt,
        custom_user_prompt='''The assistant starts with one or more cycles of (thinking about which tool to use -> performing tool call -> waiting for tool response), and ends with (thinking about the answer -> answer of the question). The thinking processes, tool calls, tool responses, and answer are enclosed within their tags. There could be multiple thinking processes, tool calls, tool call parameters and tool response parameters.

Example response:
<think> thinking process here </think>
<tool_call>
{"name": "tool name here", "arguments": {"parameter name here": parameter value here, "another parameter name here": another parameter value here, ...}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
<think> thinking process here </think>
<tool_call>
{"name": "another tool name here", "arguments": {...}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
(more thinking processes, tool calls and tool responses here)
<think> thinking process here </think>
<answer> answer here </answer>

User: '''
    )

    return bot


def app_tui():
    search_bot_dev = init_dev_search_agent_service()
    search_bot_qwen = init_qwen_agent_service()

    messages = []
    while True:
        query = input('user question: ')
        messages.append({'role': 'user', 'content': query})
        response = []
        response_plain_text = ''
        for response in search_bot_dev.run(messages=messages):
            response_plain_text = typewriter_print(response, response_plain_text)
        messages.extend(response)


def app_gui():
    agents = []
    # Original OAI-based agent configuration
    # for name, port, desc, reasoning, max_llm_calls, tools in [
    #     ('WebDancer-QwQ-32B', 8004, 'Default OAI (local)', True, 50, ['search', 'visit']),
    # ]:
    #     search_bot_dev = init_dev_search_agent_service(
    #         name=name,
    #         port=port, # For OAI
    #         desc=desc,
    #         reasoning=reasoning,
    #         max_llm_calls=max_llm_calls,
    #         tools=tools,
    #         llm_provider='oai', # Explicitly OAI
    #         llm_api_key_env='OPENAI_API_KEY' # Example if using a remote OpenAI compatible API
    #     )
    #     agents.append(search_bot_dev)

    # Add Gemini agent configuration
    # Ensure GEMINI_API_KEY is set in your environment
    gemini_agent_config = {
        'name': 'WebDancer-Gemini',
        'desc': 'Powered by Gemini',
        'reasoning': True,
        'max_llm_calls': 30, # Adjust as needed
        'tools': ['search', 'visit'],
        'llm_provider': 'gemini',
        'llm_model_name': 'gemini-1.5-flash-latest', # Or your preferred Gemini model
        'llm_api_key_env': 'GEMINI_API_KEY'
        # 'port' is not used for Gemini unless llm_model_server is specified for a proxy
    }
    try:
        gemini_bot = init_dev_search_agent_service(**gemini_agent_config)
        agents.append(gemini_bot)
    except Exception as e:
        print(f"Failed to initialize Gemini agent: {e}. Check API key and configurations.")

    # Example: Add another OAI agent if you have one, e.g. a locally deployed model
    oai_local_agent_config = {
        'name': 'WebDancer-OAI-Local',
        'port': 8004, # Example port for your local OAI compatible server
        'desc': 'Local OAI Model',
        'reasoning': True,
        'max_llm_calls': 50,
        'tools': ['search', 'visit'],
        'llm_provider': 'oai',
        'llm_api_key_env': 'LOCAL_OAI_API_KEY', # Or 'EMPTY' if not needed
        'llm_model_name': '' # Let server decide or specify if needed
    }
    try:
        oai_local_bot = init_dev_search_agent_service(**oai_local_agent_config)
        agents.append(oai_local_bot)
    except Exception as e:
        print(f"Failed to initialize local OAI agent: {e}.")


    if not agents:
        print("No agents initialized. Exiting GUI setup.")
        # Optionally, initialize a fallback or raise an error
        # For now, let's try to create a default Qwen agent if others fail
        try:
            print("Falling back to Qwen Dashscope agent.")
            qwen_bot = init_qwen_agent_service() # Ensure this function is robust
            agents.append(qwen_bot)
        except Exception as e:
            print(f"Failed to initialize Qwen fallback agent: {e}")
            print("Cannot start GUI without any active agent. Please check configurations.")
            return


    chatbot_config = {
        'prompt.suggestions': [
            '中国国足的一场比赛，国足首先失球，由一名宿姓球员扳平了。后来还发生了点球。比分最终是平均。有可能是哪几场比赛',
            'When is the paper submission deadline for the ACL 2025 Industry Track, and what is the venue address for the conference?',
            'On June 6, 2023, an article by Carolyn Collins Petersen was published in Universe Today. This article mentions a team that produced a paper about their observations, linked at the bottom of the article. Find this paper. Under what NASA award number was the work performed by R. G. Arendt supported by?',
            '有一位华语娱乐圈的重要人物，与其兄弟共同创作并主演了一部在中国南方沿海城市上映的喜剧电影，这部电影成为该类型的开山之作。与此同时，这位人物还凭借两首极具影响力的本地方言歌曲在音乐领域取得突破，极大推动了本地方言流行音乐的发展。请问，这一切发生在20世纪70年代的哪一年？',
            '有一首欧洲国家的国歌自20世纪50年代初被正式采用，并只选用了其中的一部分歌词。同一年，一位中国文艺界的重要人物创作了一部以民间传说为基础的戏曲作品，并在当年担任了多个文化领域的重要职务。请问这位中国文艺界人物是谁？',
            '有一部英国文坛上极具影响力的长篇诗歌，由一位16世纪末的著名诗人创作，这位诗人在16世纪90年代末于伦敦去世后，被安葬在一个象征英国文学传统的著名场所，与多位文学巨匠为邻。请问，这位诗人安息之地是哪里？',
            '出一份三天两夜的端午北京旅游攻略',
            '对比下最新小米汽车和保时捷性能参数，然后根据最终的结果分析下性价比最高的车型，并给出杭州的供应商',
            '量子计算突破对现有加密体系的威胁',
            '人工智能伦理框架的全球差异',
            '老龄化社会对全球养老金体系的长期冲击',
            '全球碳中和目标下的能源转型路径差异',
            '塑料污染在海洋食物链中的累积效应',
            'AI生成内容（如AI绘画）对传统艺术价值的重构'
        ],
        'user.name': 'User',
        'verbose': True
    }
    messages = {'role': 'user', 'content': '介绍下你自己'}
    WebUI(
        agent=agents,
        chatbot_config=chatbot_config,
    ).run(
        message=messages,
        share=False,
        server_name='127.0.0.1',
        server_port=7860,
        concurrency_limit=20,
        enable_mention=False,
    )


if __name__ == '__main__':
    app_gui()
