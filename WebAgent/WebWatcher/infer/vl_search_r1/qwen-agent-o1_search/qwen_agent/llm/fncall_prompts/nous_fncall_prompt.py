import copy
import json
from typing import List, Literal, Union

from qwen_agent.llm.fncall_prompts.base_fncall_prompt import BaseFnCallPrompt
from qwen_agent.llm.schema import ASSISTANT, FUNCTION, SYSTEM, USER, ContentItem, FunctionCall, Message


class NousFnCallPrompt(BaseFnCallPrompt):

    @staticmethod
    def preprocess_fncall_messages(
        messages: List[Message],
        functions: List[dict],
        lang: Literal['en', 'zh'],
        parallel_function_calls: bool = True,
        function_choice: Union[Literal['auto'], str] = 'auto',
    ) -> List[Message]:
        del lang  # ignored
        del parallel_function_calls  # ignored
        if function_choice != 'auto':
            raise NotImplementedError

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
                    fc = {'name': fn_call.name, 'arguments': json.loads(fn_call.arguments)}
                    fc = json.dumps(fc, ensure_ascii=False)
                    fc = f'<tool_call>\n{fc}\n</tool_call>'
                    content.append(ContentItem(text=fc))
                if messages[-1].role == ASSISTANT:
                    messages[-1].content.append(ContentItem(text='\n'))
                    messages[-1].content.extend(content)
                else:
                    messages.append(Message(role=role, content=content))
            elif role == FUNCTION:
                assert isinstance(content, list)
                assert len(content) == 1
                assert content[0].text
                fc = f'<tool_response>\n{content[0].text}\n</tool_response>'
                content = [ContentItem(text=fc)]
                if messages[-1].role == USER:
                    messages[-1].content.append(ContentItem(text='\n'))
                    messages[-1].content.extend(content)
                else:
                    messages.append(Message(role=USER, content=content))
            else:
                raise TypeError

        tool_descs = [{'type': 'function', 'function': f} for f in functions]
        tool_descs = '\n'.join([json.dumps(f, ensure_ascii=False) for f in tool_descs])
        tool_system = FN_CALL_TEMPLATE.format(tool_descs=tool_descs)
        if messages[0].role == SYSTEM:
            messages[0].content.append(ContentItem(text='\n\n' + tool_system))
        else:
            messages = [Message(role=SYSTEM, content=[ContentItem(text=tool_system)])] + messages
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

                i = item_text.find('<tool_call>')
                # If no function call:
                if i < 0:
                    show_text = item_text
                    if show_text:
                        new_content.append(ContentItem(text=show_text))
                    continue

                # split tool-call to separate assistant msg
                tool_call_list = item_text.split('<tool_call>')
                pre_thought = tool_call_list[0]
                if pre_thought.strip():
                    new_content.append(ContentItem(text=pre_thought))
                for txt in tool_call_list[1:]:
                    if not txt.strip():
                        continue

                    if '</tool_call>' not in txt:
                        # incomplete </tool_call>: This is to better represent incomplete tool calls in streaming output
                        fn_name, fn_args = extract_fn(txt)
                        if fn_name:  # need to call function
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
                        else:
                            show_text = remove_incomplete_special_tokens(txt)
                            if show_text:
                                new_content.append(ContentItem(text=show_text))
                        continue

                    one_tool_call_txt = txt.split('</tool_call>')

                    # The complete tool-call response
                    if new_content:
                        new_messages.append(Message(
                            role=role,
                            content=new_content,
                            extra=extra,
                        ))  # split thought and function call
                        new_content = []

                    fn = json.loads(one_tool_call_txt[0].strip())
                    new_messages.append(
                        Message(
                            role=ASSISTANT,
                            content=[],
                            function_call=FunctionCall(
                                name=fn['name'],
                                arguments=json.dumps(fn['arguments'], ensure_ascii=False),
                            ),
                            extra=extra,
                        ))

                    if one_tool_call_txt[1].strip():
                        new_content.append(ContentItem(text=one_tool_call_txt[1]))

            if new_content:
                new_messages.append(Message(role=role, content=new_content, extra=extra))
        return new_messages


FN_CALL_TEMPLATE = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_descs}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


# Mainly for removing incomplete special tokens when streaming the output
# This assumes that '<tool_call>\n{"name": "' is the special token for the NousFnCallPrompt
def remove_incomplete_special_tokens(text: str) -> str:
    if text in '<tool_call>\n{"name": "':
        text = ''
    return text


def extract_fn(text: str):
    fn_name, fn_args = '', ''
    fn_name_s = '"name": "'
    fn_name_e = '", "'
    fn_args_s = '"arguments": '
    i = text.find(fn_name_s)
    j = text.find(fn_name_e)
    k = text.find(fn_args_s)
    if i > 0:
        if j == -1:
            fn_name = text[i + len(fn_name_s):]
        else:
            fn_name = text[i + len(fn_name_s):j]
    if k > 0:
        fn_args = text[k + len(fn_args_s):]
    return fn_name, fn_args
