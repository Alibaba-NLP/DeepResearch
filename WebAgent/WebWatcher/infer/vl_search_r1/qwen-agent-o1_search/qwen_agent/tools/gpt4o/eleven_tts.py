import json
import os

import requests


def tts(text, language="en", model_id='eleven_multilingual_v2', voice_id='9BWtsMINqrJLrRacOk9x'):
    # Models: eleven_multilingual_v2 多语言; eleven_turbo_v2_5 比较快 output_format=pcm_16000 设置输出格式和采样率
    url = f'http://47.88.8.18:8088/elevenlabs/v1/text-to-speech/{voice_id}?output_format=pcm_16000'
    headers = {
        'Authorization': f'Bearer {os.environ.get("MIT_SPIDER_TOKEN")}',
        'Content-Type': 'application/json'
    }
    data = {
        "text": text,
        "model_id": model_id if language == "en" else "eleven_turbo_v2_5",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    # Add language code for Chinese TTS
    if language != "en":
        data.update({"language_code": language})
    resp = requests.post(url, headers=headers, data=json.dumps(data))
    return resp


if __name__ == '__main__':
    print(tts('你好'))
