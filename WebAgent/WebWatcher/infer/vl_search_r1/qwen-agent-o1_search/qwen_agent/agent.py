import copy
import json
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Tuple, Union

from qwen_agent.llm import get_chat_model
from qwen_agent.llm.base import BaseChatModel
from qwen_agent.llm.schema import CONTENT, DEFAULT_SYSTEM_MESSAGE, ROLE, SYSTEM, ContentItem, Message
from qwen_agent.log import logger
from qwen_agent.tools import TOOL_REGISTRY, BaseTool
from qwen_agent.utils.parallel_executor import parallel_exec
from qwen_agent.utils.utils import has_chinese_messages, merge_generate_cfgs


class Agent(ABC):
    """A base class for Agent.

    An agent can receive messages and provide response by LLM or Tools.
    Different agents have distinct workflows for processing messages and generating responses in the `_run` method.
    """

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 **kwargs):
        """Initialization the agent.

        Args:
            function_list: One list of tool name, tool configuration or Tool object,
              such as 'code_interpreter', {'name': 'code_interpreter', 'timeout': 10}, or CodeInterpreter().
            llm: The LLM model configuration or LLM model object.
              Set the configuration as {'model': '', 'api_key': '', 'model_server': ''}.
            system_message: The specified system message for LLM chat.
            name: The name of this agent.
            description: The description of this agent, which will be used for multi_agent.
        """
        if isinstance(llm, dict):
            self.llm = get_chat_model(llm)
        else:
            self.llm = llm
        self.extra_generate_cfg: dict = {}

        self.function_map = {}
        if function_list:
            print(function_list)
            for tool in function_list:
                self._init_tool(tool)

        self.system_message = system_message
        self.name = name
        self.description = description

    def run_nonstream(self, messages: List[Union[Dict, Message]], **kwargs) -> Union[List[Message], List[Dict]]:
        """Same as self.run, but with stream=False,
        meaning it returns the complete response directly
        instead of streaming the response incrementally."""
        *_, last_responses = self.run(messages, **kwargs)
        return last_responses

    def run(self, messages: List[Union[Dict, Message]],
            **kwargs) -> Union[Iterator[List[Message]], Iterator[List[Dict]]]:
        """Return one response generator based on the received messages.

        This method performs a uniform type conversion for the inputted messages,
        and calls the _run method to generate a reply.

        Args:
            messages: A list of messages.

        Yields:
            The response generator.
        """
        messages = copy.deepcopy(messages)
        _return_message_type = 'dict'
        new_messages = []
        # Only return dict when all input messages are dict
        if not messages:
            _return_message_type = 'message'
        for msg in messages:
            if isinstance(msg, dict):
                new_messages.append(Message(**msg))
            else:
                new_messages.append(msg)
                _return_message_type = 'message'

        if 'lang' not in kwargs:
            if has_chinese_messages(new_messages):
                kwargs['lang'] = 'zh'
            else:
                kwargs['lang'] = 'en'

        if self.system_message:
            if new_messages[0][ROLE] != SYSTEM:
                # Add the system instruction to the agent, default to `DEFAULT_SYSTEM_MESSAGE`
                new_messages.insert(0, Message(role=SYSTEM, content=self.system_message))
            else:
                # When the messages contain system message
                if not self.system_message.startswith(DEFAULT_SYSTEM_MESSAGE):
                    # If the user has set a special system that does not exist in messages, add
                    if isinstance(new_messages[0][CONTENT], str):
                        if not new_messages[0][CONTENT].startswith(self.system_message):
                            new_messages[0][CONTENT] = self.system_message + '\n\n' + new_messages[0][CONTENT]
                    else:
                        assert isinstance(new_messages[0][CONTENT], list)
                        assert new_messages[0][CONTENT][0].text
                        if not new_messages[0][CONTENT][0].text.startswith(self.system_message):
                            new_messages[0][CONTENT] = [ContentItem(text=self.system_message + '\n\n')
                                                       ] + new_messages[0][CONTENT]  # noqa

        for rsp in self._run(messages=new_messages, **kwargs):
            for i in range(len(rsp)):
                if not rsp[i].name and self.name:
                    rsp[i].name = self.name
            if _return_message_type == 'message':
                yield [Message(**x) if isinstance(x, dict) else x for x in rsp]
            else:
                yield [x.model_dump() if not isinstance(x, dict) else x for x in rsp]

    @abstractmethod
    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        """Return one response generator based on the received messages.

        The workflow for an agent to generate a reply.
        Each agent subclass needs to implement this method.

        Args:
            messages: A list of messages.
            lang: Language, which will be used to select the language of the prompt
              during the agent's execution process.

        Yields:
            The response generator.
        """
        raise NotImplementedError

    def run_batch(self, messages_batch: List[List[Union[Dict, Message]]], **kwargs) -> List[List[Dict]]:
        messages_batch = copy.deepcopy(messages_batch)
        lang_batch = kwargs.get('lang_batch', [])
        # _return_message_type = 'dict'
        for i, messages in enumerate(messages_batch):
            messages_batch[i] = [Message(**x) if isinstance(x, dict) else x for x in messages]
            # if messages_batch[i][0][ROLE] != SYSTEM:
            #     messages_batch[i].insert(0, Message(role=SYSTEM, content=self.system_message))
            # elif isinstance(messages_batch[i][0][CONTENT], str):
            #     messages_batch[i][0][CONTENT] = self.system_message + '\n\n' + messages_batch[i][0][CONTENT]
            # else:
            #     assert isinstance(messages_batch[i][0][CONTENT], list)
            #     messages_batch[i][0][CONTENT] = [
            #         ContentItem(text=self.system_message + '\n\n')  # noqa
            #     ] + messages_batch[i][0][CONTENT]

            # TODO: system

        if not lang_batch:
            for i, messages in enumerate(messages_batch):
                if has_chinese_messages(messages):
                    lang_batch.append('zh')
                else:
                    lang_batch.append('en')
            kwargs['lang_batch'] = lang_batch
        assert len(lang_batch) == len(messages_batch)

        responses_batch = self._run_batch(messages_batch=messages_batch, **kwargs)
        responses_batch = [
            [x.model_dump() if not isinstance(x, dict) else x for x in responses] for responses in responses_batch
        ]
        return responses_batch

    def _run_batch(self, messages_batch: List[List[Union[Dict, Message]]], lang_batch: List[str],
                   **kwargs) -> List[List[Message]]:

        def _ask_run(index: int, messages: List[Message], lang: str = 'en') -> tuple:
            *_, last = self._run(messages=messages, lang=lang)
            return index, last

        data = [{
            'index': i,
            'messages': messages,
            'lang': lang
        } for i, (messages, lang) in enumerate(zip(messages_batch, lang_batch))]
        results = parallel_exec(_ask_run, data, max_workers=20, jitter=0.5)
        ordered_results = sorted(results, key=lambda x: x[0])
        return [x[-1] for x in ordered_results]

    def _call_llm(
        self,
        messages: List[Message],
        functions: Optional[List[Dict]] = None,
        stream: bool = True,
        extra_generate_cfg: Optional[dict] = None,
    ) -> Iterator[List[Message]]:
        """The interface of calling LLM for the agent.

        We prepend the system_message of this agent to the messages, and call LLM.

        Args:
            messages: A list of messages.
            functions: The list of functions provided to LLM.
            stream: LLM streaming output or non-streaming output.
              For consistency, we default to using streaming output across all agents.

        Yields:
            The response generator of LLM.
        """
        return self.llm.chat(messages=messages,
                             functions=functions,
                             stream=stream,
                             extra_generate_cfg=merge_generate_cfgs(
                                 base_generate_cfg=self.extra_generate_cfg,
                                 new_generate_cfg=extra_generate_cfg,
                             ))

    def _call_tool(self, tool_name: str, tool_args: Union[str, dict] = '{}', **kwargs) -> Union[str, List[ContentItem]]:
        """The interface of calling tools for the agent.

        Args:
            tool_name: The name of one tool.
            tool_args: Model generated or user given tool parameters.

        Returns:
            The output of tools.
        """
        if tool_name not in self.function_map:
            return f'Tool {tool_name} does not exists.'
        tool = self.function_map[tool_name]
        try:
            tool_result = tool.call(tool_args, **kwargs)
        except (CIServiceError, DocParserError) as ex:
            raise ex
        except Exception as ex:
            # Need to raise tool error
            exception_message = str(ex)

            # logger.info(exception_message)
            # raise ToolServiceError(code='400', message='Tool Execution Failed.')

            exception_type = type(ex).__name__
            traceback_info = ''.join(traceback.format_tb(ex.__traceback__))
            error_message = f'[call_tool_error] An error occurred when calling tool `{tool_name}`:\n' \
                            f'{exception_type}: {exception_message}'
                            # f'Traceback:\n{traceback_info}'
            logger.warning(error_message)
            return error_message
        except BaseException as ex:
            # Need to raise tool error
            exception_message = str(ex)
            exception_type = type(ex).__name__
            traceback_info = ''.join(traceback.format_tb(ex.__traceback__))
            error_message = f'[FATAL] An error occurred when calling tool `{tool_name}`:\n' \
                            f'{exception_type}: {exception_message}'
                            # f'Traceback:\n{traceback_info}'
            logger.warning(error_message)
            return error_message

        if isinstance(tool_result, str):
            return tool_result
        elif isinstance(tool_result, list) and all(isinstance(item, ContentItem) for item in tool_result):
            return tool_result  # multimodal tool results
        else:
            return json.dumps(tool_result, ensure_ascii=False, indent=4)

    def _init_tool(self, tool: Union[str, Dict, BaseTool]):
        if isinstance(tool, BaseTool):
            tool_name = tool.name
            if tool_name in self.function_map:
                logger.warning(f'Repeatedly adding tool {tool_name}, will use the newest tool in function list')
            self.function_map[tool_name] = tool
        else:
            if isinstance(tool, dict):
                tool_name = tool['name']
                tool_cfg = tool
            else:
                tool_name = tool
                tool_cfg = None
            if tool_name not in TOOL_REGISTRY:
                raise ValueError(f'Tool {tool_name} is not registered.')

            if tool_name in self.function_map:
                logger.warning(f'Repeatedly adding tool {tool_name}, will use the newest tool in function list')
            self.function_map[tool_name] = TOOL_REGISTRY[tool_name](tool_cfg)

    def _detect_tool(self, message: Message) -> Tuple[bool, str, str, str]:
        """A built-in tool call detection for func_call format message.

        Args:
            message: one message generated by LLM.

        Returns:
            Need to call tool or not, tool name, tool args, text replies.
        """
        func_name = None
        func_args = None

        if message.function_call:
            func_call = message.function_call
            func_name = func_call.name
            func_args = func_call.arguments
        text = message.content
        if not text:
            text = ''

        return (func_name is not None), func_name, func_args, text


# The most basic form of an agent is just a LLM, not augmented with any tool or workflow.
class BasicAgent(Agent):

    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        extra_generate_cfg = {'lang': lang}
        if kwargs.get('seed') is not None:
            extra_generate_cfg['seed'] = kwargs['seed']
        return self._call_llm(messages, extra_generate_cfg=extra_generate_cfg)
