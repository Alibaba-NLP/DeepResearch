import copy
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from multiprocessing import Lock

from tqdm.auto import tqdm

from logger import Logger
from openai_style_api_client import OpenAIAPIClient
from utils import load_file2list, get_args, APIException

LLM_CALL_SLEEP = 2
LOCK = Lock()


def update_call_args(data, call_args):
    remove_keys = list()
    call_args = vars(args)
    for key in call_args.keys():
        if call_args.get(key) is None:
            remove_keys.append(key)
    for key in remove_keys:
        call_args.pop(key)
    assert not ('messages' in data.get("call_args", {}) and data.get('prompt') not in [None, ''])
    system = None
    if call_args.get('system') and len(call_args['system']) > 0:
        system = call_args['system']
    if 'messages' not in data.get("call_args", {}):
        data['call_args'] = {}
        messages = []
        if len(data.get('system', '')) > 0:
            messages.append({"role": "system", "content": data['system']})
        elif system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": data['prompt']})
        data['call_args']["messages"] = messages
    elif system and 'messages' in data.get("call_args", {}):
        messages = data['call_args']['messages']
        if messages[0]['role'] != 'system':
            messages.insert(0, {'role': 'system', 'content': args.system})
    call_args.update(data['call_args'])
    if "call_args" in data:
        call_args.update(data['call_args'])
    data['call_args'] = call_args
    return data


def llm_api_wrapper(api_call, data):
    try:
        ret = api_call(**data['call_args'])
    except APIException as e:
        print(''.join(traceback.format_exception(*sys.exc_info())))
        time.sleep(LLM_CALL_SLEEP)
        return None
    if ret is not None:
        try:
            gen_l = list()
            key = 'messages'
            if isinstance(ret['choices'], list):
                if key not in ret['choices']:
                    key = 'message'
                for choice in ret['choices']:
                    if isinstance(choice[key], list):
                        gen_l.append(choice[key][-1]['content'])
                    else:
                        gen_l.append(choice[key]['content'])
            data.update({'gen': gen_l})
            return data
        except Exception as e:
            print(''.join(traceback.format_exception(*sys.exc_info())))
            return None
    else:
        time.sleep(LLM_CALL_SLEEP)
        return None


def llm_api_generate(args, api_call):
    in_file = args.pop('in_file', None)
    out_file = args.pop('out_file', None)
    if not os.path.exists(in_file):
        Logger.error(f"input file: {in_file} not found!!!")
    ori_data_l = load_file2list(in_file)
    if len(ori_data_l) == 0:
        Logger.info(f'{in_file} is empty.')
        return
    oai_data_l = list()
    for data in ori_data_l:
        oai_data = update_call_args(data, args)
        oai_data_l.append(copy.deepcopy(oai_data))
    if args.pop('local_cache', True):
        write_mode = 'a+'
        local_key_s = set()
        for data in load_file2list(out_file):
            local_key_s.add(data['uuid'])
        oai_data_l = [d for d in oai_data_l if d['uuid'] not in local_key_s]
    else:
        write_mode = 'w'
    if len(oai_data_l) == 0:
        return
    failed_count = 0
    llm_api_wrapper_partial = partial(llm_api_wrapper, api_call=api_call)
    with open(out_file, write_mode, encoding='utf-8') as writer, ThreadPoolExecutor(args.num_workers) as executor:
        Logger.info(f"gen answer uuids: {' ,'.join([str(d['uuid']) for d in oai_data_l[:6]])}, ...")
        future_l = [executor.submit(llm_api_wrapper_partial, data=data) for data in oai_data_l]
        with tqdm(total=len(oai_data_l), desc=f"LLM Infer[{args.num_workers}]", ncols=80) as progress:
            for future in as_completed(future_l):
                progress.update(1)
                try:
                    result = future.result()
                except Exception as e:
                    result = None
                if not result:
                    failed_count += 1
                    Logger.red(f"failed_count: {failed_count}")
                    continue
                LOCK.acquire()
                writer.write(json.dumps(result, ensure_ascii=False) + '\n')
                writer.flush()
                LOCK.release()


if __name__ == '__main__':
    args = get_args()
    args.pop('pre')
    args.pop('completion')
    api = OpenAIAPIClient(call_url=args.pop("call_url", None),
                          api_key=args.pop("api_key", None))
    llm_api_generate(args, api.call)
