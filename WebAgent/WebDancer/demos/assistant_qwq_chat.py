"""An image generation agent implemented by assistant with qwq"""

import os

from qwen_agent.agents import Assistant
from qwen_agent.utils.output_beautify import typewriter_print

from demos.agents.search_agent import SearchAgent
from demos.llm.oai import TextChatAtOAI
from demos.llm.qwen_dashscope import QwenChatAtDS
from demos.gui.web_ui import WebUI
from demos.utils.date import date2str, get_date_now
from demos.tools import Visit, Search


ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')



def init_dev_search_agent_service(name: str = 'SEARCH', port: int = 8002, desc: str = '初版', reasoning: bool = True, max_llm_calls: int = 20, tools = ['search', 'visit'], addtional_agent = None):
    llm_cfg = TextChatAtOAI({
        'model': '',
        'model_type': 'oai',
        'model_server': f'http://127.0.0.1:{port}/v1',
        'api_key': 'EMPTY',
        'generate_cfg': {
            'fncall_prompt_type': 'nous',
            'temperature': 0.6,
            'top_p': 0.95,
            'top_k': -1,
            'repetition_penalty': 1.1,
            'max_tokens': 32768,
            'stream_options': {
                'include_usage': True,
            },
            'timeout': 3000
        },
    })
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



def app_gui():
    agents = []
    for name, port, desc, reasoning, max_llm_calls, tools in [
        ('WebDancer-QwQ-32B', 8004, '...', True, 50, ['search', 'visit']),
    ]:
        search_bot_dev = init_dev_search_agent_service(
            name=name,
            port=port,
            desc=desc,
            reasoning=reasoning,
            max_llm_calls=max_llm_calls,
            tools=tools,
        )
        agents.append(search_bot_dev)


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
