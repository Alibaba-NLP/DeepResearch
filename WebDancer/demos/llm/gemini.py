import os
import copy
import json
import logging
from http import HTTPStatus
from pprint import pformat
from typing import Dict, Iterator, List, Optional, Literal, Union

import google.generativeai as genai
# Ensure genai.types is imported correctly for Tool, FunctionDeclaration etc.
from google.generativeai import types as genai_types

from qwen_agent.llm.base import ModelServiceError, register_llm
from qwen_agent.llm.function_calling import BaseFnCallModel, simulate_response_completion_with_chat
from qwen_agent.llm.schema import ASSISTANT, USER, SYSTEM, TOOL, Message, FunctionCall
from qwen_agent.log import logger


@register_llm('gemini')
class TextChatAtGemini(BaseFnCallModel):

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        cfg = cfg or {}

        self.model_name = self.model or cfg.get('model') or 'gemini-1.5-flash'  # Default model

        api_key = cfg.get('api_key') or os.getenv('GEMINI_API_KEY')
        if not api_key:
            # Defer raising error to actual API call time, or log a warning
            logger.warning("GEMINI_API_KEY is not set. API calls will fail.")
            self._api_key_configured = False
        else:
            genai.configure(api_key=api_key)
            self._api_key_configured = True

        # Initialize the generative model client
        # We will create the chat session per request, or use generate_content directly
        self.client = genai.GenerativeModel(self.model_name)


        # Default generation config, can be overridden by generate_cfg in methods
        self.default_generate_cfg = {
            "temperature": cfg.get('temperature', 0.7),
            "top_p": cfg.get('top_p', None), # Gemini doesn't always require top_p
            "top_k": cfg.get('top_k', None), # Gemini doesn't always require top_k
            "max_output_tokens": cfg.get('max_tokens', 8192), # Gemini 1.5 Flash has a large context
            "candidate_count": cfg.get('n', 1), # Number of candidate responses
        }
        # Merge with any specific generate_cfg from main config
        self.default_generate_cfg.update(cfg.get('generate_cfg', {}))
        # Remove None values from default_generate_cfg as Gemini SDK expects them to be absent if not set
        self.default_generate_cfg = {k: v for k, v in self.default_generate_cfg.items() if v is not None}


    @staticmethod
    def _convert_messages_to_gemini_format(
        messages: List[Message],
        system_message_content: Optional[str] = None # Explicit system message override
    ) -> (List[genai_types.ContentDict], Optional[str]): # Returns history and system_instruction string
        gemini_history_contents = []
        # System instruction for Gemini's generate_content is a simple string
        extracted_system_instruction_str = system_message_content

        # Iterate through messages to build history and extract system instruction if not overridden
        for msg in messages:
            role = msg.role
            parts = []

            if role == SYSTEM:
                if msg.content and not extracted_system_instruction_str: # Only use if not already overridden
                    extracted_system_instruction_str = msg.content
                # System messages are not part of Gemini's direct 'contents' history,
                # they are passed as a separate 'system_instruction' parameter.
                continue

            if role == ASSISTANT:
                role = 'model'
                if msg.content: # Ensure content is not None or empty before adding
                    parts.append(genai_types.PartDict(text=msg.content))
                if msg.function_call: # Assistant requests a function call
                    try:
                        # Arguments should be a dict for Gemini's FunctionCallDict
                        args = json.loads(msg.function_call.arguments or '{}')
                        if not isinstance(args, dict): # Ensure it's a dict after loading
                            logger.error(f"Parsed function call arguments is not a dict: {args}")
                            args = {"_raw_args": msg.function_call.arguments} # Fallback
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse function call arguments: {msg.function_call.arguments}")
                        args = {"_raw_args": msg.function_call.arguments} # Fallback
                    parts.append(genai_types.PartDict(
                        function_call=genai_types.FunctionCallDict(
                            name=msg.function_call.name,
                            args=args
                        )
                    ))
            elif role == USER:
                role = 'user'
                if msg.content: # Ensure content is not None or empty
                    parts.append(genai_types.PartDict(text=msg.content))
            elif role == TOOL: # This is a function response from our agent
                role = 'function' # Gemini uses 'function' role for function responses

                # The 'response' in FunctionResponseDict should be a dict.
                # msg.content from qwen_agent is typically a JSON string of the tool's output.
                try:
                    tool_response_data = json.loads(msg.content)
                    if not isinstance(tool_response_data, dict):
                        # If the JSON is not an object, Gemini might expect it wrapped, e.g. {"result": tool_response_data}
                        # For now, let's ensure it's a dict. If it's just a string, wrap it.
                        tool_response_data = {"content": tool_response_data}
                except json.JSONDecodeError:
                    # If content is not a valid JSON string, treat it as plain text result.
                    tool_response_data = {"content": msg.content}
                except TypeError: # Handle if msg.content is None
                    tool_response_data = {"content": ""}


                parts.append(genai_types.PartDict(
                    function_response=genai_types.FunctionResponseDict(
                        name=msg.name, # Name of the function that was called
                        response=tool_response_data # This should be a dict
                    )
                ))
            else:
                logger.warning(f"Unknown role {msg.role} in message conversion. Skipping message: {msg}")
                continue

            if parts: # Only add if there are parts to send
                gemini_history_contents.append(genai_types.ContentDict(role=role, parts=parts))
            elif role == 'model' and not msg.function_call : # Model message must have content or function call
                 logger.debug(f"Model message from qwen_agent has no content and no function_call: {msg}. Adding empty text part.")
                 # Add an empty text part if it's a model message with no content and no FC, to maintain turn structure
                 gemini_history_contents.append(genai_types.ContentDict(role=role, parts=[genai_types.PartDict(text='')]))
            elif role == 'user' and not msg.content: # User message usually has content
                 logger.debug(f"User message from qwen_agent has no content: {msg}. Adding empty text part if no other parts.")
                 gemini_history_contents.append(genai_types.ContentDict(role=role, parts=[genai_types.PartDict(text='')]))


        return gemini_history_contents, extracted_system_instruction_str


    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        if not self._api_key_configured:
            raise ModelServiceError(message="GEMINI_API_KEY is not configured.")

        current_generate_cfg_dict = {**self.default_generate_cfg, **generate_cfg}

        tools_for_request = current_generate_cfg_dict.pop('tools', None)
        explicit_system_instruction_in_cfg = current_generate_cfg_dict.pop('system_instruction', None)

        gemini_history, system_instruction_str = self._convert_messages_to_gemini_format(
            messages, system_message_content=explicit_system_instruction_in_cfg
        )

        gemini_sdk_generation_config = genai_types.GenerationConfig(**current_generate_cfg_dict)

        logger.debug(f"Gemini Stream Request History: {pformat(gemini_history)}")
        logger.debug(f"Gemini Stream Request System Instruction: {system_instruction_str}")
        logger.debug(f"Gemini Stream Request SDK Config: {pformat(gemini_sdk_generation_config)}")
        logger.debug(f"Gemini Stream Request Tools: {pformat(tools_for_request)}")

        try:
            response_stream = self.client.generate_content(
                contents=gemini_history,
                generation_config=gemini_sdk_generation_config,
                stream=True,
                tools=tools_for_request,
                system_instruction=system_instruction_str
            )

            accumulated_content = ""
            for chunk in response_stream:
                # Check for blocked prompt/response first
                if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                    logger.error(f"Gemini stream blocked due to prompt: {chunk.prompt_feedback.block_reason_message}")
                    raise ModelServiceError(message=f"Content submission/generation blocked by Gemini: {chunk.prompt_feedback.block_reason_message}")

                # Each chunk can have candidates, and each candidate can have content with parts
                if not (hasattr(chunk, 'candidates') and chunk.candidates):
                    logger.debug(f"Gemini stream chunk has no candidates: {chunk}")
                    continue # Or handle as error if this state is unexpected

                # Assuming we are interested in the first candidate
                candidate = chunk.candidates[0]
                if not (hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts):
                    # Check for finish_reason if content is empty (e.g. safety block on response)
                    if hasattr(candidate, 'finish_reason') and candidate.finish_reason.name not in ['STOP', 'MAX_TOKENS', 'UNSPECIFIED']: # Other reasons might be blocks
                         logger.error(f"Gemini stream candidate finished with reason: {candidate.finish_reason.name}. Message: {getattr(candidate, 'finish_message', '')}")
                         # Specific safety ratings check can be added here if needed
                         if candidate.finish_reason.name == 'SAFETY':
                             raise ModelServiceError(message=f"Content generation stopped by Gemini due to safety settings. {getattr(candidate, 'finish_message', '')}")
                    logger.debug(f"Gemini stream chunk candidate has no content or parts: {candidate}")
                    continue

                # Process parts (usually one primary part, but can be multiple for mixed content)
                chunk_text_this_iteration = ""
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        # Gemini usually sends the whole FC at once.
                        qwen_fn_call = FunctionCall(name=fc.name, arguments=json.dumps(fc.args or {}))
                        yield [Message(role=ASSISTANT, content='', function_call=qwen_fn_call)]
                        return # Stop further processing for this turn

                    if hasattr(part, 'text'):
                        chunk_text_this_iteration += part.text

                if not chunk_text_this_iteration:
                    logger.debug("Stream chunk yielded no text this iteration.")
                    continue

                if delta_stream:
                    yield [Message(role=ASSISTANT, content=chunk_text_this_iteration)]
                else:
                    accumulated_content += chunk_text_this_iteration
                    yield [Message(role=ASSISTANT, content=accumulated_content)]

        except genai.types.BlockedPromptException as e:
            logger.error(f"Gemini stream request blocked by API: {e}")
            raise ModelServiceError(message=f"Content submission blocked by Gemini: {e}", exception=e)
        except Exception as e:
            logger.error(f"Error during Gemini stream: {type(e).__name__}: {e}")
            raise ModelServiceError(exception=e)


    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict,
    ) -> List[Message]:
        if not self._api_key_configured:
            raise ModelServiceError(message="GEMINI_API_KEY is not configured.")

        current_generate_cfg_dict = {**self.default_generate_cfg, **generate_cfg}
        tools_for_request = current_generate_cfg_dict.pop('tools', None)
        explicit_system_instruction_in_cfg = current_generate_cfg_dict.pop('system_instruction', None)


        gemini_history, system_instruction_str = self._convert_messages_to_gemini_format(
            messages, system_message_content=explicit_system_instruction_in_cfg
        )

        gemini_sdk_generation_config = genai_types.GenerationConfig(**current_generate_cfg_dict)

        logger.debug(f"Gemini Non-Stream Request History: {pformat(gemini_history)}")
        logger.debug(f"Gemini Non-Stream Request System Instruction: {system_instruction_str}")
        logger.debug(f"Gemini Non-Stream Request SDK Config: {pformat(gemini_sdk_generation_config)}")
        logger.debug(f"Gemini Non-Stream Request Tools: {pformat(tools_for_request)}")

        try:
            response = self.client.generate_content(
                contents=gemini_history,
                generation_config=gemini_sdk_generation_config,
                stream=False,
                tools=tools_for_request,
                system_instruction=system_instruction_str
            )

            # Check for prompt feedback (e.g. blocked prompt)
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                logger.error(f"Gemini non-stream request blocked due to prompt: {response.prompt_feedback.block_reason_message}")
                raise ModelServiceError(message=f"Content submission blocked by Gemini: {response.prompt_feedback.block_reason_message}")

            if not (hasattr(response, 'candidates') and response.candidates):
                 logger.error(f"Gemini response has no candidates: {response}")
                 raise ModelServiceError(message="Gemini returned no candidates in response.")

            candidate = response.candidates[0]
            # Check for finish_reason if content is empty (e.g. safety block on response)
            if not (hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts):
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason.name not in ['STOP', 'MAX_TOKENS', 'UNSPECIFIED']:
                     logger.error(f"Gemini non-stream candidate finished with reason: {candidate.finish_reason.name}. Message: {getattr(candidate, 'finish_message', '')}")
                     if candidate.finish_reason.name == 'SAFETY':
                         raise ModelServiceError(message=f"Content generation stopped by Gemini due to safety settings. {getattr(candidate, 'finish_message', '')}")
                     raise ModelServiceError(message=f"Gemini generation stopped due to: {candidate.finish_reason.name}. {getattr(candidate, 'finish_message', '')}")
                logger.error(f"Gemini response candidate has no content or parts: {candidate}")
                return [Message(role=ASSISTANT, content='')] # Or raise error

            # Process parts
            response_text_parts = []
            for part in candidate.content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    qwen_fn_call = FunctionCall(name=fc.name, arguments=json.dumps(fc.args or {}))
                    return [Message(role=ASSISTANT, content='', function_call=qwen_fn_call)]

                if hasattr(part, 'text'):
                    response_text_parts.append(part.text)

            final_text = "".join(response_text_parts)

            # Fallback to response.text if parts didn't yield text (though less likely with above logic)
            if not final_text and not any(hasattr(p, 'function_call') for p in candidate.content.parts):
                try:
                    final_text = response.text # This consolidates text from all parts of first candidate
                except ValueError: # If .text is not available (e.g. only function call, already handled)
                    logger.warning("response.text not available and no function call found. Returning empty.")
                    final_text = ""

            return [Message(role=ASSISTANT, content=final_text)]

        except genai.types.BlockedPromptException as e:
            logger.error(f"Gemini non-stream request blocked by API: {e}")
            raise ModelServiceError(message=f"Content submission blocked by Gemini: {e}", exception=e)
        except Exception as e:
            logger.error(f"Error during Gemini non-stream call: {type(e).__name__}: {e}")
            if hasattr(e, 'message') and isinstance(e.message, str):
                raise ModelServiceError(message=e.message, exception=e)
            raise ModelServiceError(exception=e)


    @staticmethod
    def _convert_functions_to_gemini_tools(functions: List[Dict]) -> Optional[List[genai_types.Tool]]:
        if not functions:
            return None

        gemini_function_declarations = []
        for func_def in functions:
            openapi_schema_params = func_def.get('parameters', None)
            if openapi_schema_params:
                if 'properties' in openapi_schema_params and 'type' not in openapi_schema_params:
                    openapi_schema_params['type'] = 'object'
            else:
                openapi_schema_params = None # Explicitly None if no parameters

            try:
                # Ensure parameters is a dict if not None, or None itself.
                # The genai SDK might be strict about this.
                # An empty dict {} is valid for no parameters if parameters cannot be None.
                # Based on FunctionDeclaration, parameters can be mapping or None.
                # If openapi_schema_params is an empty dict and should be None, adjust here.
                # For now, assuming func_def['parameters'] provides a valid OpenAPI schema dict or is absent.

                declaration = genai_types.FunctionDeclaration(
                    name=func_def['name'],
                    description=func_def.get('description', ''),
                    parameters=openapi_schema_params
                )
                gemini_function_declarations.append(declaration)
            except Exception as e:
                logger.error(f"Failed to create Gemini FunctionDeclaration for {func_def.get('name')}: {e}. Schema: {openapi_schema_params}")
                continue # Skip problematic function

        if not gemini_function_declarations:
            return None
        return [genai_types.Tool(function_declarations=gemini_function_declarations)]

    def _chat_with_functions(
        self,
        messages: List[Message],
        functions: List[Dict],
        stream: bool,
        delta_stream: bool,
        generate_cfg: dict,
        lang: Literal['en', 'zh'],
    ) -> Union[List[Message], Iterator[List[Message]]]:

        # Start with a copy of default_generate_cfg, then update with specific generate_cfg
        # This ensures that 'tools' or 'system_instruction' from default_generate_cfg aren't incorrectly propagated
        # if they are meant to be context-specific from the current call.
        current_api_call_cfg = {**self.default_generate_cfg, **generate_cfg}

        # Convert qwen_agent function definitions to Gemini's Tool format
        gemini_tools = self._convert_functions_to_gemini_tools(functions)

        # The 'tools' parameter is passed directly to generate_content, not as part of GenerationConfig object.
        # So, we'll add it to current_api_call_cfg to be extracted by _chat_stream/_chat_no_stream.
        if gemini_tools:
            current_api_call_cfg['tools'] = gemini_tools
        else: # Ensure 'tools' is not present if no functions, to avoid sending empty tools list if that's an issue.
            current_api_call_cfg.pop('tools', None)

        # Clean up any non-Gemini SDK parameters from the portion that will become GenerationConfig
        # 'tools' and 'system_instruction' are handled at the top level of generate_content call.
        keys_to_remove_from_sdk_generation_config_obj = [
            'fncall_prompt_type', 'parallel_function_calls',
            'function_choice', 'thought_in_content', 'stream_options',
            'tools', 'system_instruction' # These are handled by chat methods directly
        ]
        final_sdk_config_params = {
            k: v for k, v in current_api_call_cfg.items()
            if k not in keys_to_remove_from_sdk_generation_config_obj and k in genai_types.GenerationConfig.__annotations__
        }
        # We pass the full current_api_call_cfg (which includes 'tools' and potentially 'system_instruction' at top level)
        # to the chat methods, which then construct the GenerationConfig object from relevant parts and pass 'tools' etc separately.

        # messages are already List[Message]
        # simulate_response_completion_with_chat is not used for Gemini as it has native FC support.

        if stream:
            return self._chat_stream(messages, delta_stream=delta_stream, generate_cfg=current_api_call_cfg)
        else:
            return self._chat_no_stream(messages, generate_cfg=current_api_call_cfg)

    def _chat(
        self,
        messages: List[Union[Message, Dict]],
        stream: bool,
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Union[List[Message], Iterator[List[Message]]]:
        processed_messages: List[Message] = []
        for msg_item in messages:
            if isinstance(msg_item, dict):
                try:
                    processed_messages.append(Message(**msg_item))
                except Exception as e:
                    logger.error(f"Failed to convert dict to Message in _chat: {msg_item}. Error: {e}")
                    continue
            elif isinstance(msg_item, Message):
                processed_messages.append(msg_item)
            else:
                logger.error(f"Invalid message type in _chat: {type(msg_item)}. Expected Message or Dict.")
                continue

        # This generate_cfg might come from _chat_with_functions (containing 'tools')
        # or from a direct call to .chat() (without 'tools' if no functions intended).
        # _chat_stream and _chat_no_stream are responsible for correctly using 'tools' and other
        # relevant parts from generate_cfg to call Gemini's generate_content.
        if stream:
            return self._chat_stream(processed_messages, delta_stream=delta_stream, generate_cfg=generate_cfg)
        else:
            return self._chat_no_stream(processed_messages, generate_cfg=generate_cfg)
