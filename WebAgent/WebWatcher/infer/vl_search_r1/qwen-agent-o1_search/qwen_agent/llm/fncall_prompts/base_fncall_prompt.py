from typing import List, Literal, Union

from qwen_agent.llm.schema import FUNCTION, Message
from qwen_agent.utils.utils import format_as_multimodal_message, format_as_text_message, has_chinese_messages


class BaseFnCallPrompt(object):

    @staticmethod
    def preprocess_fncall_messages(
        messages: List[Message],
        functions: List[dict],
        lang: Literal['en', 'zh'],
        parallel_function_calls: bool = True,
        function_choice: Union[Literal['auto'], str] = 'auto',
    ) -> List[Message]:
        """
        Preprocesss the messages and add the function calling prompt,
        assuming the input and output messages are in the multimodal format.
        """
        assert function_choice != 'none'
        raise NotImplementedError

    @staticmethod
    def postprocess_fncall_messages(
        messages: List[Message],
        parallel_function_calls: bool = True,
        function_choice: Union[Literal['auto'], str] = 'auto',
    ) -> List[Message]:
        """
        Transform the plaintext model output into structured function call messages,
        return in the multimodal format for consistency.
        """
        raise NotImplementedError

    def format_plaintext_train_samples(
        self,
        messages: List[Union[Message, dict]],
        functions: List[dict],
        lang: Literal['auto', 'en', 'zh'] = 'auto',
        parallel_function_calls: bool = True,
    ) -> List[Message]:
        messages = [m if isinstance(m, Message) else Message(**m) for m in messages]

        if lang == 'auto':
            lang = 'zh' if has_chinese_messages(messages) else 'en'

        if not parallel_function_calls:
            for i in range(len(messages) - 1):
                has_para = (messages[i].function_call and messages[i + 1].function_call)
                has_para = has_para or ((messages[i].role == FUNCTION) and (messages[i + 1].role == FUNCTION))
                if has_para:
                    raise ValueError('This sample requires parallel_function_calls=True.')

        messages = [
            format_as_multimodal_message(msg, add_upload_info=True, add_multimodel_upload_info=True, lang=lang)
            for msg in messages
        ]
        for m in messages:
            for item in m.content:
                if item.type != 'text':
                    raise NotImplementedError('Support for multimodal samples not implemented yet.')

        messages = self.preprocess_fncall_messages(
            messages=messages,
            functions=functions,
            lang=lang,
            parallel_function_calls=parallel_function_calls,
        )

        messages = [format_as_text_message(msg, add_upload_info=False) for msg in messages]
        return messages
