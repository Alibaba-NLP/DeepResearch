import urllib.parse
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
import re

def process_url(url, sub_url):
    """
    Args:
        url (str): url
        sub_url (str): sub_url
    
    Returns:
        str: the processed url
    """
    return urllib.parse.urljoin(url, sub_url)


def clean_markdown(res):
    """
    Args:
        res (str): markdown content
    
    Returns:
        str: cleaned markdown content
    """
    pattern = r'\[.*?\]\(.*?\)'
    try:
        result = re.sub(pattern, '', res)
        url_pattern = pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        result = re.sub(url_pattern, '', result)
        result = result.replace("* \n","")
        result = re.sub(r"\n\n+", "\n", result)
        return result
    except Exception:
        return res

async def get_info(url, screenshot = True) -> str:
    """
    Args:
        url (str): url
        screentshot (bool): whether to take a screenshot
    
    Returns:
        str: html content and cleaned markdown content
    """
    run_config = CrawlerRunConfig(
        screenshot=True,             # Grab a screenshot as base64
        screenshot_wait_for=1.0,     # Wait 1s before capturing
    )
    async with AsyncWebCrawler() as crawler:
        if screenshot:
            result = await crawler.arun(url, config=run_config)
            return result.html, clean_markdown(result.markdown), result.screenshot
        else:
            result = await crawler.arun(url, screenshot=screenshot)
            return result.html, clean_markdown(result.markdown)
    
def get_content_between_a_b(start_tag, end_tag, text):
    """
    Args:
        start_tag (str): start_tag
        end_tag (str): end_tag
        text (str): complete sentence

    Returns:
        str: the content between start_tag and end_tag
    """
    extracted_text = ""
    start_index = text.find(start_tag)
    while start_index != -1:
        end_index = text.find(end_tag, start_index + len(start_tag))
        if end_index != -1:
            extracted_text += text[start_index + len(start_tag) : end_index] + " "
            start_index = text.find(start_tag, end_index + len(end_tag))
        else:
            break

    return extracted_text.strip()
