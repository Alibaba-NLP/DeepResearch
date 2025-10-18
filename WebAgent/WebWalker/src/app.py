import streamlit as st
import os
import json5
from agent import WebWalker
from qwen_agent.tools.base import BaseTool, register_tool
import os
import re
import json
import asyncio
from utils import *
import base64
from PIL import Image
from bs4 import BeautifulSoup



if 'DASHSCOPE_API_KEY' not in os.environ and 'OPENAI_API_KEY' not in os.environ:
    raise ValueError("Please set the environment variable 'DASHSCOPE_API_KEY' or 'OPENAI_API_KEY' to your API key.")
if 'DASHSCOPE_API_KEY' in os.environ:
    model = "qwen-plus"
    llm_cfg = {
        'model': model,
        'api_key': os.getenv('DASHSCOPE_API_KEY'),
        'model_server': "https://dashscope.aliyuncs.com/compatible-mode/v1" ,
        'generate_cfg': {
                'top_p': 0.8,
                'max_input_tokens': 120000,
                'max_retries': 20
        },
    }
if 'OPENAI_API_KEY' in os.environ and 'OPENAI_MODEL_SERVER' in os.environ:
    model = "gpt-4o"
    llm_cfg = {
        'model': model,
        'api_key': os.getenv('OPENAI_API_KEY'),
        'model_server': os.getenv('OPENAI_MODEL_SERVER'),
        'generate_cfg': {
                'top_p': 0.8,
                'max_input_tokens': 120000,
                'max_retries': 20
        },
    }
"""
llm_cfg = {
    'model': model,
    'api_key': api_key,
    'model_server': model_server,
    'generate_cfg': {
            'top_p': 0.8,
            'max_input_tokens': 120000,
            'max_retries': 20
    },
}
"""

def extract_links_with_text(html):
    """
    Args:
        html (str): html content
    
    Returns:
        str: clickable buttons
    """
    with open("ROOT_URL.txt", "r") as f:
        ROOT_URL = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    links = []

    for a_tag in soup.find_all('a', href=True):
        url = a_tag['href']
        text = ''.join(a_tag.stripped_strings)
        
        if text and "javascript" not in url and not url.endswith(('.jpg', '.png', '.gif', '.jpeg', '.pdf')):
            if process_url(ROOT_URL, url).startswith(ROOT_URL):
                links.append({'url': process_url(ROOT_URL, url), 'text': text})

    for a_tag in soup.find_all('a', onclick=True):
        onclick_text = a_tag['onclick']
        text = ''.join(a_tag.stripped_strings)
        
        match = re.search(r"window\.location\.href='([^']*)'", onclick_text)
        if match:
            url = match.group(1)
            if url and text  and not url.endswith(('.jpg', '.png', '.gif', '.jpeg', '.pdf')):
                if process_url(ROOT_URL, url).startswith(ROOT_URL):
                    links.append({'url': process_url(ROOT_URL, url), 'text': text})

    for a_tag in soup.find_all('a', attrs={'data-url': True}):
        url = a_tag['data-url']
        text = ''.join(a_tag.stripped_strings)
        if url and text and not url.endswith(('.jpg', '.png', '.gif', '.jpeg', '.pdf')):
            if process_url(ROOT_URL, url).startswith(ROOT_URL):
                links.append({'url': process_url(ROOT_URL, url), 'text': text})

    for a_tag in soup.find_all('a', class_='herf-mask'):
        url = a_tag.get('href')
        text = a_tag.get('title') or ''.join(a_tag.stripped_strings)
        if url and text and not url.endswith(('.jpg', '.png', '.gif', '.jpeg', '.pdf')):
            if process_url(ROOT_URL, url).startswith(ROOT_URL):
                    links.append({'url': process_url(ROOT_URL, url), 'text': text})

    for button in soup.find_all('button', onclick=True):
        onclick_text = button['onclick']
        text = button.get('title') or button.get('aria-label') or ''.join(button.stripped_strings)
        match = re.search(r"window\.location\.href='([^']*)'", onclick_text)
        if match:
            url = match.group(1)
            if url and text:
                if process_url(ROOT_URL, url).startswith(ROOT_URL):
                    links.append({'url': process_url(ROOT_URL, url), 'text': text})

    unique_links = {f"{item['url']}_{item['text']}": item for item in links}  # 去重

    if not os.path.exists("BUTTON_URL_ADIC.json"):
        with open("BUTTON_URL_ADIC.json", "w") as f:
            json.dump({}, f)
    with open("BUTTON_URL_ADIC.json", "r") as f:
        BUTTON_URL_ADIC = json.load(f)
    for temp in list(unique_links.values()):
        BUTTON_URL_ADIC[temp["text"]] = temp["url"]
    with open("BUTTON_URL_ADIC.json", "w") as f:
        json.dump(BUTTON_URL_ADIC, f)
    info = ""
    for i in list(unique_links.values()):
        info += "<button>" + i["text"] + "<button>" + "\n"
    return info

if __name__ == "__main__":
    st.title('🤝WebWalker')
    st.markdown("### 📚Introduction")
    st.markdown("👋Welcome to WebWalker! WebWalker is a web-based conversational agent that can help you navigate websites and find information.")
    st.markdown("📑The paper of WebWalker is available at [arXiv]().")
    st.markdown("✨You can bulid your own WebWalker by following the [instruction](https://github.com/Alibaba-NLP/WebWalker).")
    st.markdown("🙋If you have any questions, please feel free to contact us via the [Github issue](https://github.com/Alibaba-NLP/WebWalker/issue).")
    st.markdown("### 🚀Let's start exploring the website!")
    if 'form_1_text' not in st.session_state:
        st.session_state.form_1_text = ""
    if 'form_2_text' not in st.session_state:
        st.session_state.form_2_text = ""

    with st.sidebar:
        MAX_ROUNDS = st.number_input('Max Count Count：', min_value=1, max_value=15, value=10, step=1)
        website_example = st.sidebar.selectbox('Example Website：', ['https://2025.aclweb.org/'])
        question_example = st.sidebar.selectbox('Example Query：', ["When is the paper submission deadline for ACL 2025 Industry Track, and what is the venue specific address?", "Who is the general chair of ACL 2025?", "What is the spcial theme of ACL 2025?"])

    col1, col2 = st.columns([3, 1]) 
    with col1:
        with st.form(key='my_form'):
            form_1_text = st.text_area("**🤯Memory**", value="No Memory", height=68)
            website = st.text_area('👉Website', value=website_example, placeholder='Input the website you want to walk through.')
            query = st.text_area('🤔Query', value=question_example, placeholder='Input the query you want to ask.')
            submit_button = st.form_submit_button('Start!!!!')
            
            if submit_button:
                if website and query:
                    tools = ["visit_page"]  
                    llm_cfg["query"] = query
                    llm_cfg["action_count"] = MAX_ROUNDS
                    bot = WebWalker(llm=llm_cfg,
                        function_list=tools
                        )
                    BUTTON_URL_ADIC = {}
                    ROOT_URL = website
                    with open("ROOT_URL.txt", "w") as f:
                        f.write(ROOT_URL)
                    messages = []  # This stores the chat history.
                    visited_links = []
                    start_prompt = "query:\n{query} \nofficial website:\n{website}".format(query=query, website=website)
                    st.markdown('**🌐Now visit**')
                    st.write(website)
                    html, markdown, screenshot = asyncio.run(get_info(website))
                    with col2:
                        st.markdown('**📸Observation**')
                        if screenshot:
                            st.session_state.image_index = 0
                            print("get screenshot!")
                            image_folder = "images/"
                            if not os.path.exists(image_folder):
                                os.makedirs(image_folder)
                            with open(image_folder+str(st.session_state.image_index)+".png", "wb") as f:
                                f.write(base64.b64decode(screenshot))
                            image_files = os.listdir(image_folder)
                            image_files = [f for f in image_files if f.endswith(('.png', '.jpg', '.jpeg'))]
                            image_path = os.path.join(image_folder, image_files[st.session_state.image_index])
                            image = Image.open(image_folder+str(st.session_state.image_index)+".png")
                            st.image(image, caption='Start Obervation', width=400)
                        
                    buttons = extract_links_with_text(html)
                    response = "website information:\n\n" + markdown + "\n\n"
                    response += "clickable button:\n\n" + buttons + "\n\nEach button is wrapped in a <button> tag"
                    start_prompt += "\nObservation:" + response + "\n\n"
                    messages.append({'role': 'user', 'content':start_prompt})
                    cnt = 0
                    response = []
                    response = bot.run(messages=messages, lang = "zh")
                    for i in response:
                        if "\"}" in i[0]["content"] and "Memory" not in i[0]["content"]:
                            st.markdown('**💭Thoughts**')
                            st.markdown(i[0]["content"].split("Action")[0])
                        elif "\"}" in i[0]["content"] and "Memory" in i[0]["content"]:
                            st.text_area('**🤯Memory Update**', i[0]["content"][:-2])
                        if "Final Answer" in i[0]["content"]:
                            st.session_state.answer = i[0]["content"]
                            st.markdown('**🙋Anwser**')
                            st.write(st.session_state.answer)
                else:
                    st.error('Please input the website and query.')


@register_tool('visit_page',allow_overwrite=True)
class VisitPage(BaseTool):
    """
    description: A tool that visits a webpage and extracts the content of the page.
    parameters:
        - name: url
            type: string
            description: The URL of the webpage to visit.
            required: true
    """
    description = 'A tool analyzes the content of a webpage and extracts buttons associated with sublinks. Simply input the button which you want to explore, and the tool will return both the markdown-formatted content of the corresponding page of button and a list of new clickable buttons found on the new page.'
    parameters = [{
        'name': 'button',
        'type': 'string',
        'description': 'the button you want to click',
        'required': True
    }]

            
    def call(self, params: str, **kwargs) -> str:
        if not params.strip().endswith("}"):
            if "}" in params.strip():
                params = "{" + get_content_between_a_b("{","}",params) + "}"
            else:
                if not params.strip().endswith("\""):
                    params = params.strip() + "\"}"
                else:
                    params = params.strip() + "}"
        params = "{" + get_content_between_a_b("{","}",params) + "}"
        if 'button' in json5.loads(params):
            with open("BUTTON_URL_ADIC.json", "r") as f:
                BUTTON_URL_ADIC = json.load(f)
            if json5.loads(params)['button'].replace("<button>","") in BUTTON_URL_ADIC:
                st.markdown('**👆Click Button**')
                st.write(json5.loads(params)['button'].replace("<button>",""))
                url =  BUTTON_URL_ADIC[json5.loads(params)['button'].replace("<button>","")]
                html, markdown, screenshot = asyncio.run(get_info(url))
                st.markdown('**🌐Now Visit**')
                st.write(url)
                with col2:
                    st.write("")
                    st.markdown('**📸Observation**')
                    if screenshot:
                        print("get screenshot!")
                        image_folder = "images/"
                        with open(image_folder+str(st.session_state.image_index+1)+".png", "wb") as f:
                            f.write(base64.b64decode(screenshot))
                        st.session_state.image_index += 1
                        image = Image.open(image_folder+str(st.session_state.image_index)+".png")
                        st.image(image, caption='Step ' + str(st.session_state.image_index) + ' Obervation', width=400)
                response_bottons = extract_links_with_text(html)
                response_content = markdown
                if response_content:
                    response = "The web informtaion if:\n\n" + response_content + "\n\n"
                else:
                    response = "The information of the current page is not accessible\n\n"
                response += "Clickable buttons are wrapped in <button> tag" + response_bottons
                return response
            else:
                return "The button can not be clicked, please retry a new botton!"
        else:
            return "Your input is invalid, plase output the action input correctly!"
