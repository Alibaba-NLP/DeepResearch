import base64
import copy
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import sys
import time
import traceback
import urllib.parse
from io import BytesIO
from typing import Any, List, Literal, Optional, Tuple, Union

import json5
import requests
from pydantic import BaseModel

from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, FUNCTION, SYSTEM, USER, ContentItem, Message
from qwen_agent.log import logger


def append_signal_handler(sig, handler):
    """
    Installs a new signal handler while preserving any existing handler.
    If an existing handler is present, it will be called _after_ the new handler.
    """

    old_handler = signal.getsignal(sig)
    if not callable(old_handler):
        old_handler = None
        if sig == signal.SIGINT:

            def old_handler(*args, **kwargs):
                raise KeyboardInterrupt
        elif sig == signal.SIGTERM:

            def old_handler(*args, **kwargs):
                raise SystemExit

    def new_handler(*args, **kwargs):
        handler(*args, **kwargs)
        if old_handler is not None:
            old_handler(*args, **kwargs)

    signal.signal(sig, new_handler)


def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


def hash_sha256(text: str) -> str:
    hash_object = hashlib.sha256(text.encode())
    key = hash_object.hexdigest()
    return key


def print_traceback(is_error: bool = True):
    tb = ''.join(traceback.format_exception(*sys.exc_info(), limit=3))
    if is_error:
        logger.error(tb)
    else:
        logger.warning(tb)


CHINESE_CHAR_RE = re.compile(r'[\u4e00-\u9fff]')


def has_chinese_chars(data: Any) -> bool:
    text = f'{data}'
    return bool(CHINESE_CHAR_RE.search(text))


def has_chinese_messages(messages: List[Union[Message, dict]], check_roles: Tuple[str] = (SYSTEM, USER)) -> bool:
    for m in messages:
        if m['role'] in check_roles:
            if has_chinese_chars(m['content']):
                return True
    return False


def get_basename_from_url(path_or_url: str, need_rm_uuid: bool = False) -> str:
    if re.match(r'^[A-Za-z]:\\', path_or_url):
        # "C:\\a\\b\\c" -> "C:/a/b/c"
        path_or_url = path_or_url.replace('\\', '/')

    # "/mnt/a/b/c" -> "c"
    # "https://github.com/here?k=v" -> "here"
    # "https://github.com/" -> ""
    basename = urllib.parse.urlparse(path_or_url).path
    basename = os.path.basename(basename)
    basename = urllib.parse.unquote(basename)
    basename = basename.strip()

    # "https://github.com/" -> "" -> "github.com"
    if not basename:
        basename = [x.strip() for x in path_or_url.split('/') if x.strip()][-1]

    new_basename = basename
    if need_rm_uuid:
        try:
            # Hotfix: rm uuid
            if len(basename) > 38 and basename[8] == '-' and basename[13] == '-' and basename[18] == '-' and basename[
                    23] == '-' and basename[36] == '_':
                new_basename = basename[37:]
        except Exception:
            new_basename = basename
    return new_basename


def is_http_url(path_or_url: str) -> bool:
    if path_or_url.startswith('https://') or path_or_url.startswith('http://'):
        return True
    return False


def is_image(path_or_url: str) -> bool:
    filename = get_basename_from_url(path_or_url).lower()
    for ext in ['jpg', 'jpeg', 'png', 'webp']:
        if filename.endswith(ext):
            return True
    return False


def sanitize_chrome_file_path(file_path: str) -> str:
    if os.path.exists(file_path):
        return file_path

    # Dealing with "file:///...":
    new_path = urllib.parse.urlparse(file_path)
    new_path = urllib.parse.unquote(new_path.path)
    new_path = sanitize_windows_file_path(new_path)
    if os.path.exists(new_path):
        return new_path

    return sanitize_windows_file_path(file_path)


def sanitize_windows_file_path(file_path: str) -> str:
    # For Linux and macOS.
    if os.path.exists(file_path):
        return file_path

    # For native Windows, drop the leading '/' in '/C:/'
    win_path = file_path
    if win_path.startswith('/'):
        win_path = win_path[1:]
    if os.path.exists(win_path):
        return win_path

    # For Windows + WSL.
    if re.match(r'^[A-Za-z]:/', win_path):
        wsl_path = f'/mnt/{win_path[0].lower()}/{win_path[3:]}'
        if os.path.exists(wsl_path):
            return wsl_path

    # For native Windows, replace / with \.
    win_path = win_path.replace('/', '\\')
    if os.path.exists(win_path):
        return win_path

    return file_path


def save_url_to_local_work_dir(url: str, save_dir: str, save_filename: str = '') -> str:
    if not save_filename:
        save_filename = get_basename_from_url(url)
    new_path = os.path.join(save_dir, save_filename)
    if os.path.exists(new_path):
        os.remove(new_path)
    # logger.info(f'Downloading {url} to {new_path}...')
    start_time = time.time()
    if not is_http_url(url):
        url = sanitize_chrome_file_path(url)
        shutil.copy(url, new_path)
    else:
        headers = {
            'User-Agent':
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            with open(new_path, 'wb') as file:
                file.write(response.content)
        else:
            raise ValueError('Can not download this file. Please check your network or the file link.')
    end_time = time.time()
    # logger.info(f'Finished downloading {url} to {new_path}. Time spent: {end_time - start_time} seconds.')
    return new_path


def save_text_to_file(path: str, text: str) -> None:
    with open(path, 'w', encoding='utf-8') as fp:
        fp.write(text)


def read_text_from_file(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8') as file:
            file_content = file.read()
    except UnicodeDecodeError:
        print_traceback(is_error=False)
        from charset_normalizer import from_path
        results = from_path(path)
        file_content = str(results.best())
    return file_content


def contains_html_tags(text: str) -> bool:
    pattern = r'<(p|span|div|li|html|script)[^>]*?'
    return bool(re.search(pattern, text))


def get_content_type_by_head_request(path: str) -> str:
    try:
        response = requests.head(path, timeout=5)
        content_type = response.headers.get('Content-Type', '')
        return content_type
    except requests.RequestException:
        return 'unk'


def get_file_type(path: str) -> Literal['pdf', 'docx', 'pptx', 'csv', 'tsv', 'xlsx', 'xls','zip','mp3','jsonl','pdb','py','xml']:
    f_type = get_basename_from_url(path).split('.')[-1].lower()
    if is_image(path):
        return "image"
    if f_type in ['pdf', 'docx', 'pptx', 'csv', 'tsv', 'xlsx', 'xls','zip','mp3','jsonl','pdb','py','xml']:
        # Specially supported file types
        return f_type

    if is_http_url(path):
        # The HTTP header information for the response is obtained by making a HEAD request to the target URL,
        # where the Content-type field usually indicates the Type of Content to be returned
        content_type = get_content_type_by_head_request(path)
        if 'application/pdf' in content_type:
            return 'pdf'
        elif 'application/msword' in content_type:
            return 'docx'

        # Assuming that the URL is HTML by default,
        # because the file downloaded by the request may contain html tags
        return 'html'
    else:
        # Determine by reading local HTML file
        try:
            content = read_text_from_file(path)
        except Exception:
            print_traceback()
            return 'unk'

        if contains_html_tags(content):
            return 'html'
        else:
            return 'txt'


def extract_urls(text: str) -> List[str]:
    pattern = re.compile(r'https?://\S+')
    urls = re.findall(pattern, text)
    return urls


def extract_markdown_urls(md_text: str) -> List[str]:
    pattern = r'!?\[[^\]]*\]\(([^\)]+)\)'
    urls = re.findall(pattern, md_text)
    return urls


def extract_code(text: str) -> str:
    # Match triple backtick blocks first
    triple_match = re.search(r'```[^\n]*\n(.+?)```', text, re.DOTALL)
    if triple_match:
        text = triple_match.group(1)
    else:
        try:
            text = json5.loads(text)['code']
        except Exception:
            print_traceback(is_error=False)
    # If no code blocks found, return original text
    return text


def json_loads(text: str) -> dict:
    text = text.strip('\n')
    if text.startswith('```') and text.endswith('\n```'):
        text = '\n'.join(text.split('\n')[1:-1])
    try:
        return json.loads(text)
    except json.decoder.JSONDecodeError as json_err:
        try:
            return json5.loads(text)
        except ValueError:
            raise json_err


class PydanticJSONEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        return super().default(obj)


def json_dumps_pretty(obj: dict, ensure_ascii=False, indent=2, **kwargs) -> str:
    return json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent, cls=PydanticJSONEncoder, **kwargs)


def json_dumps_compact(obj: dict, ensure_ascii=False, indent=None, **kwargs) -> str:
    return json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent, cls=PydanticJSONEncoder, **kwargs)


def format_as_multimodal_message(
    msg: Message,
    add_upload_info: bool,
    add_multimodel_upload_info: bool,
    lang: Literal['auto', 'en', 'zh'] = 'auto',
) -> Message:
    assert msg.role in (USER, ASSISTANT, SYSTEM, FUNCTION)
    content: List[ContentItem] = []
    if isinstance(msg.content, str):  # if text content
        if msg.content:
            content = [ContentItem(text=msg.content)]
    elif isinstance(msg.content, list):  # if multimodal content
        files = []
        for item in msg.content:
            k, v = item.get_type_and_value()
            if k in ('text', 'image', 'audio', 'video'):
                content.append(item)
            if k == 'file':
                # Move 'file' out of 'content' since it's not natively supported by models
                files.append((v, k))
            if add_multimodel_upload_info and k in ('image', 'video'):
                # Indicate the image name
                # TODO: considering audio
                files.append((v, k))
        if add_upload_info and files and (msg.role in (SYSTEM, USER)):
            if lang == 'auto':
                has_zh = has_chinese_chars(msg)
            else:
                has_zh = (lang == 'zh')
            upload = []
            # for f, k in [(get_basename_from_url(f, need_rm_uuid=True), k) for f, k in files]:
            for f, k in [(f, k) for f, k in files]:
                if k == 'image':
                    if has_zh:
                        upload.append(f'![图片]({f})')
                    else:
                        upload.append(f'![image]({f})')
                elif k == 'video':
                    if has_zh:
                        upload.append(f'![视频]({f})')
                    else:
                        upload.append(f'![video]({f})')
                else:
                    if has_zh:
                        upload.append(f'[文件]({f})')
                    else:
                        upload.append(f'[file]({f})')
            upload = ' '.join(upload)
            if has_zh:
                upload = f'（上传了 {upload}）\n\n'
            else:
                upload = f'(Uploaded {upload})\n\n'

            # Check and avoid adding duplicate upload info
            upload_info_already_added = False
            for item in content:
                if item.text and (upload in item.text):
                    upload_info_already_added = True

            if not upload_info_already_added:
                content = [ContentItem(text=upload)] + content
    else:
        raise TypeError
    msg = Message(role=msg.role,
                  content=content,
                  name=msg.name if msg.role == FUNCTION else None,
                  function_call=msg.function_call,
                  extra=msg.extra)
    return msg


def format_as_text_message(
    msg: Message,
    add_upload_info: bool,
    lang: Literal['auto', 'en', 'zh'] = 'auto',
) -> Message:
    msg = format_as_multimodal_message(msg,
                                       add_upload_info=add_upload_info,
                                       add_multimodel_upload_info=add_upload_info,
                                       lang=lang)
    text = ''
    for item in msg.content:
        if item.type == 'text':
            text += item.value
    msg.content = text
    return msg


def extract_text_from_message(
    msg: Message,
    add_upload_info: bool,
    lang: Literal['auto', 'en', 'zh'] = 'auto',
) -> str:
    if isinstance(msg.content, list):
        text = format_as_text_message(msg, add_upload_info=add_upload_info, lang=lang).content
    elif isinstance(msg.content, str):
        text = msg.content
    else:
        raise TypeError(f'List of str or str expected, but received {type(msg.content).__name__}.')
    return text.strip()


def extract_files_from_messages(messages: List[Message], include_images: bool) -> List[str]:
    files = []
    for msg in messages:
        if isinstance(msg.content, list):
            for item in msg.content:
                if item.file and item.file not in files:
                    files.append(item.file)
                if include_images and item.image and item.image not in files:
                    files.append(item.image)
    return files


def merge_generate_cfgs(base_generate_cfg: Optional[dict], new_generate_cfg: Optional[dict]) -> dict:
    generate_cfg: dict = copy.deepcopy(base_generate_cfg or {})
    if new_generate_cfg:
        for k, v in new_generate_cfg.items():
            if k == 'stop':
                stop = generate_cfg.get('stop', [])
                stop = stop + [s for s in v if s not in stop]
                generate_cfg['stop'] = stop
            else:
                generate_cfg[k] = v
    return generate_cfg


def build_text_completion_prompt(
    messages: List[Message],
    allow_special: bool = False,
    default_system: str = DEFAULT_SYSTEM_MESSAGE,
) -> str:
    im_start = '<|im_start|>'
    im_end = '<|im_end|>'

    if messages[0].role == SYSTEM:
        sys = messages[0].content
        assert isinstance(sys, str)
        prompt = f'{im_start}{SYSTEM}\n{sys}{im_end}'
        messages = messages[1:]
    else:
        prompt = f'{im_start}{SYSTEM}\n{default_system}{im_end}'

    # Make sure we are completing the chat in the tone of the assistant
    if messages[-1].role != ASSISTANT:
        messages = messages + [Message(ASSISTANT, '')]

    for msg in messages:
        assert isinstance(msg.content, str)
        content = msg.content
        if allow_special:
            assert msg.role in (USER, ASSISTANT, SYSTEM, FUNCTION)
            if msg.function_call:
                assert msg.role == ASSISTANT
                tool_call = msg.function_call.arguments
                try:
                    tool_call = {'name': msg.function_call.name, 'arguments': json.loads(tool_call)}
                    tool_call = json.dumps(tool_call, ensure_ascii=False, indent=2)
                except json.decoder.JSONDecodeError:
                    tool_call = '{"name": "' + msg.function_call.name + '", "arguments": ' + tool_call + '}'
                if content:
                    content += '\n'
                content += f'<tool_call>\n{tool_call}\n</tool_call>'
        else:
            assert msg.role in (USER, ASSISTANT)
            assert msg.function_call is None
        prompt += f'\n{im_start}{msg.role}\n{content}{im_end}'

    assert prompt.endswith(im_end)
    prompt = prompt[:-len(im_end)]
    return prompt


def encode_image_as_base64(path: str, max_short_side_length: int = -1) -> str:
    from PIL import Image
    image = Image.open(path)

    if (max_short_side_length > 0) and (min(image.size) > max_short_side_length):
        ori_size = image.size
        image = resize_image(image, short_side_length=max_short_side_length)
        logger.debug(f'Image "{path}" resized from {ori_size} to {image.size}.')

    image = image.convert(mode='RGB')
    buffered = BytesIO()
    image.save(buffered, format='JPEG')
    return 'data:image/jpeg;base64,' + base64.b64encode(buffered.getvalue()).decode('utf-8')


def load_image_from_base64(image_base64: Union[bytes, str]):
    from PIL import Image
    image = Image.open(BytesIO(base64.b64decode(image_base64)))
    image.load()
    return image


def resize_image(img, short_side_length: int = 1080):
    from PIL import Image
    assert isinstance(img, Image.Image)

    width, height = img.size

    if width <= height:
        new_width = short_side_length
        new_height = int((short_side_length / width) * height)
    else:
        new_height = short_side_length
        new_width = int((short_side_length / height) * width)

    resized_img = img.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)
    return resized_img


def get_last_usr_msg_idx(messages: List[Union[dict, Message]]) -> int:
    i = len(messages) - 1
    while (i >= 0) and (messages[i]['role'] != 'user'):
        i -= 1
    assert i >= 0, messages
    assert messages[i]['role'] == 'user'
    return i
