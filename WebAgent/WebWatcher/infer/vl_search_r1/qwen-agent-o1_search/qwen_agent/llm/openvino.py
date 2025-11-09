import copy
from pprint import pformat
from threading import Thread
from typing import Dict, Iterator, List, Optional

from qwen_agent.llm.base import register_llm
from qwen_agent.llm.function_calling import BaseFnCallModel
from qwen_agent.llm.schema import ASSISTANT, Message
from qwen_agent.log import logger
from qwen_agent.utils.utils import build_text_completion_prompt


@register_llm('openvino')
class OpenVINO(BaseFnCallModel):
    """
    OpenVINO Pipeline API.

    To use, you should have the 'optimum[openvino]' python package installed.

    Example export and quantize openvino model by command line:
        optimum-cli export openvino --model Qwen/Qwen2-7B-Instruct --task text-generation-with-past --weight-format int4 --group-size 128 --ratio 0.8 Qwen2-7B-Instruct-ov

    Example passing pipeline in directly:
        llm_cfg = {
            'ov_model_dir': 'Qwen2-7B-Instruct-ov',
            'model_type': 'openvino',
            'device': 'cpu'
            }
        system_instruction = '''You are a helpful assistant.
        After receiving the user's request, you should:
        - first draw an image and obtain the image url,
        - then run code `request.get(image_url)` to download the image,
        - and finally select an image operation from the given document to process the image.
        Please show the image using `plt.show()`.'''
        tools = ['my_image_gen', 'code_interpreter']
        files = ['./examples/resource/doc.pdf']
        bot = Assistant(llm=llm_cfg,
                system_message=system_instruction,
                function_list=tools,
                files=files)
    """

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        if 'ov_model_dir' not in cfg:
            raise ValueError('Please provide openvino model directory through `ov_model_dir` in cfg.')

        try:
            from optimum.intel.openvino import OVModelForCausalLM
        except ImportError as e:
            raise ImportError('Could not import optimum-intel python package for openvino. '
                              'Please install it with: '
                              "pip install -U 'optimum[openvino]'") from e
        try:
            from transformers import AutoConfig, AutoTokenizer
        except ImportError as e:
            raise ImportError('Could not import transformers python package for openvino. '
                              'Please install it with: '
                              "pip install -U 'transformers'") from e

        self.ov_model = OVModelForCausalLM.from_pretrained(
            cfg['ov_model_dir'],
            device=cfg.get('device', 'cpu'),
            ov_config=cfg.get('ov_config', {}),
            config=AutoConfig.from_pretrained(cfg['ov_model_dir']),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg['ov_model_dir'])

    def _get_stopping_criteria(self, generate_cfg: dict):
        from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

        class StopSequenceCriteria(StoppingCriteria):
            """
            This class can be used to stop generation whenever a sequence of tokens is encountered.

            Args:
                stop_sequences (`str` or `List[str]`):
                    The sequence (or list of sequences) on which to stop execution.
                tokenizer:
                    The tokenizer used to decode the model outputs.
            """

            def __init__(self, stop_sequences, tokenizer):
                if isinstance(stop_sequences, str):
                    stop_sequences = [stop_sequences]
                self.stop_sequences = stop_sequences
                self.tokenizer = tokenizer

            def __call__(self, input_ids, scores, **kwargs) -> bool:
                decoded_output = self.tokenizer.decode(input_ids.tolist()[0])
                return any(decoded_output.endswith(stop_sequence) for stop_sequence in self.stop_sequences)

        return StoppingCriteriaList([StopSequenceCriteria(generate_cfg['stop'], self.tokenizer)])

    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        from transformers import TextIteratorStreamer
        generate_cfg = copy.deepcopy(generate_cfg)
        prompt = build_text_completion_prompt(messages)
        logger.debug(f'LLM Input:\n{pformat(prompt, indent=2)}')
        input_token = self.tokenizer(prompt, return_tensors='pt').input_ids
        streamer = TextIteratorStreamer(self.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
        generate_cfg.update(
            dict(
                input_ids=input_token,
                streamer=streamer,
                max_new_tokens=generate_cfg.get('max_new_tokens', 2048),
                stopping_criteria=self._get_stopping_criteria(generate_cfg=generate_cfg),
            ))
        del generate_cfg['stop']

        def generate_and_signal_complete():
            self.ov_model.generate(**generate_cfg)

        t1 = Thread(target=generate_and_signal_complete)
        t1.start()
        partial_text = ''
        for new_text in streamer:
            partial_text += new_text
            if delta_stream:
                yield [Message(ASSISTANT, new_text)]
            else:
                yield [Message(ASSISTANT, partial_text)]

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict,
    ) -> List[Message]:
        generate_cfg = copy.deepcopy(generate_cfg)
        prompt = build_text_completion_prompt(messages)
        logger.debug(f'LLM Input:\n{pformat(prompt, indent=2)}')
        input_token = self.tokenizer(prompt, return_tensors='pt').input_ids
        generate_cfg.update(
            dict(
                input_ids=input_token,
                max_new_tokens=generate_cfg.get('max_new_tokens', 2048),
                stopping_criteria=self._get_stopping_criteria(generate_cfg=generate_cfg),
            ))
        del generate_cfg['stop']
        response = self.ov_model.generate(**generate_cfg)
        response = response[:, len(input_token[0]):]
        answer = self.tokenizer.batch_decode(response, skip_special_tokens=True)[0]
        return [Message(ASSISTANT, answer)]
