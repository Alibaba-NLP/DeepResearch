import copy
import json
import os
from typing import List, Literal, Union

from qwen_agent.llm.fncall_prompts.base_fncall_prompt import BaseFnCallPrompt
from qwen_agent.llm.schema import ASSISTANT, FUNCTION, SYSTEM, USER, ContentItem, FunctionCall, Message


class CodeFnCallPrompt(BaseFnCallPrompt):

    @staticmethod
    def preprocess_fncall_messages(
        messages: List[Message],
        functions: List[dict],
        lang: Literal['en', 'zh'],
        parallel_function_calls: bool = True,
        function_choice: Union[Literal['auto'], str] = 'auto',
    ) -> List[Message]:
        del lang  # ignored
        assert not parallel_function_calls  # ignored
        if function_choice != 'auto':
            raise NotImplementedError

        assert len(functions) == 1

        ori_messages = messages

        # Change function_call responses to plaintext responses:
        messages = []
        for msg in copy.deepcopy(ori_messages):
            role, content = msg.role, msg.content
            if role in (SYSTEM, USER):
                messages.append(msg)
            elif role == ASSISTANT:
                content = (content or [])
                fn_call = msg.function_call
                if fn_call:
                    fc = json.loads(fn_call.arguments)['code']
                    fc = f'{FN_START}{fc}{FN_END}'
                    content.append(ContentItem(text=fc))
                if messages[-1].role == ASSISTANT:
                    messages[-1].content.extend(content)
                else:
                    messages.append(Message(role=role, content=content))
            elif role == FUNCTION:
                assert isinstance(content, list)
                assert len(content) == 1
                assert content[0].text
                fc = f'{OBS_START}\n{content[0].text}\n{OBS_END}'
                content = [ContentItem(text=fc)]
                assert messages[-1].role == ASSISTANT
                messages[-1].content.extend(content)
            else:
                raise TypeError

        return messages

    @staticmethod
    def postprocess_fncall_messages(
        messages: List[Message],
        parallel_function_calls: bool = True,
        function_choice: Union[Literal['auto'], str] = 'auto',
    ) -> List[Message]:
        if function_choice != 'auto':
            raise NotImplementedError

        # Convert plaintext responses to function_call responses:
        new_messages = []
        for msg in messages:
            role, content, extra = msg.role, msg.content, msg.extra
            assert isinstance(content, list)

            if role in (SYSTEM, USER):
                new_messages.append(Message(role=role, content=content, extra=extra))
                continue

            new_content = []
            for item in content:
                item_type, item_text = item.get_type_and_value()

                if item_type != 'text':  # multimodal
                    new_content.append(item)
                    continue

                i = item_text.find(FN_START)
                # If no function call:
                if i < 0:
                    # Remove incomplete FN_START
                    show_text = remove_incomplete_special_tokens(item_text)
                    if show_text:
                        new_content.append(ContentItem(text=show_text))
                    continue

                # split tool-call to separate assistant msg
                tool_call_list = item_text.split(FN_START)
                pre_thought = tool_call_list[0]
                if pre_thought.strip():
                    new_content.append(ContentItem(text=pre_thought))
                for txt in tool_call_list[1:]:
                    if not txt.strip():
                        continue

                    if FN_END not in txt:
                        fn_name, fn_args = DEFAULT_FN_NAME, json.dumps(
                            {'code': remove_incomplete_special_tokens_for_fn(txt)}, ensure_ascii=False)
                        if new_content:
                            new_messages.append(Message(
                                role=role,
                                content=new_content,
                                extra=extra,
                            ))  # split thought and function call
                            new_content = []
                        new_messages.append(
                            Message(
                                role=ASSISTANT,
                                content=[],
                                function_call=FunctionCall(
                                    name=fn_name,
                                    arguments=fn_args,
                                ),
                                extra=extra,
                            ))

                        continue

                    one_tool_call_txt = txt.split(FN_END)

                    # The complete tool-call response
                    if new_content:
                        new_messages.append(Message(
                            role=role,
                            content=new_content,
                            extra=extra,
                        ))  # split thought and function call
                        new_content = []

                    fn = json.dumps({'code': one_tool_call_txt[0]}, ensure_ascii=False)
                    new_messages.append(
                        Message(
                            role=ASSISTANT,
                            content=[],
                            function_call=FunctionCall(
                                name=DEFAULT_FN_NAME,
                                arguments=fn,
                            ),
                            extra=extra,
                        ))

                    if one_tool_call_txt[1].strip():
                        new_content.append(ContentItem(text=one_tool_call_txt[1]))

            if new_content:
                new_messages.append(Message(role=role, content=new_content, extra=extra))
        return new_messages


FN_START = '```python\n'
FN_END = '\n```\n'
OBS_START = '```output'
OBS_END = '```\n'
DEFAULT_FN_NAME = 'code_interpreter_http'

if int(os.getenv('ENABLE_EXEC_TOOL', '1')):
    FN_STOP_WORDS = [OBS_START]  # If the tool is actually called
else:
    FN_STOP_WORDS = []  # If we assume the tool results


# Mainly for removing incomplete special tokens when streaming the output
# This assumes that '\n```python' is the special token
def remove_incomplete_special_tokens(text: str) -> str:
    if text.endswith('```'):
        text = text[:-len('```')]
    elif text.endswith('```python'):
        text = text[:-len('```python')]
    return text


# This assumes that '\n```\n' is the special token
def remove_incomplete_special_tokens_for_fn(text: str) -> str:
    if text.endswith('\n'):
        text = text[:-len('\n')]
    if text.endswith('\n``'):
        text = text[:-len('\n``')]
    return text


def extract_fn(text: str):
    fn_name, fn_args = DEFAULT_FN_NAME, 'text'
    return fn_name, fn_args
