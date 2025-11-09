import torch
import re
import numpy as np
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
from transformers.image_processing_base import BatchFeature
from PIL import Image
from tqdm import tqdm
import json
from .qwen_tool_call import Qwen_agent
import concurrent.futures


def process_image(image, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))
    elif isinstance(image, str):
        image = Image.open(image)


    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    # logging: dict
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3

class LLMGenerationManager:
    def __init__(
        self,
        processor,
        actor_rollout_wg,
        config: GenerationConfig,
        # logger: Tracking,
        is_validation: bool = False,
        text_input: bool=False
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        # self.logger = logger
        self.is_validation = is_validation
        self.text_input = text_input

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))
        self.qwen_agent = Qwen_agent(function_list=['web_search','VLSearchImage','visit'])

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']
    
    def _postprocess_responses_first(self,batch):
        
        responses_str = self.tokenizer.batch_decode(batch.batch['input_ids'], skip_special_tokens=True)
        responses_str = ["<search>"+item.split('Question: ')[1].split(' \n\nassistant\n')[0]+"</search>" for item in responses_str]

        responses = self._batch_tokenize(responses_str)
        return responses, responses_str
        

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )
        def extract_tags(text):
            # 定义正则表达式，匹配 <answer>...</answer>、<search>...</search> 和 <think>...</think>
            pattern = r"<(answer|think|tool_call)>(.*?)</\1>"
            # 使用 findall 方法找到所有匹配的内容
            matches = re.findall(pattern, text, re.DOTALL)
            # 将匹配的内容重新组合成字符串
            result = "\n".join([f"<{tag}>{content}</{tag}>" for tag, content in matches])
            return result

        responses_str = [extract_tags(resp) + self.tokenizer.eos_token for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List, rollings) -> torch.Tensor:
        """Process next observations from environment."""
        # len([item for item in next_obs if isinstance(item, dict) and item['tool'] not in ['VLSearchImage','VLSearchText','web_search']])
        next_obs_str = []
        multi_modal_data = []
        multi_modal_inputs = []
        merge_length = self.processor.image_processor.merge_size**2
        # print(self.retrievaled_images)
        for idx, obs_item in enumerate(next_obs):
            if isinstance(obs_item,str):
                next_obs_str.append(obs_item)
                multi_modal_data.append({'image': []})
                multi_modal_inputs.append(BatchFeature(dict()))
            elif obs_item['status'] == False:
                next_obs_str.append('\n<|im_start|>user\nYour previous action is invalid.\n<|im_end|>\n<|im_start|>assistant\n')
                multi_modal_data.append({'image': []})
                multi_modal_inputs.append(BatchFeature(dict()))
            elif obs_item['status'] == True:
                text = False
                if text:
                    results = obs_item['result']
                    obs_str = '\n<|im_start|>user\n' + results + '<|im_end|>\n<|im_start|>assistant\n'
                    next_obs_str.append(obs_str)
                    multi_modal_data.append({'image': []})
                    multi_modal_inputs.append(BatchFeature(dict()))
                else:
                    if obs_item['tool'] in ['VLSearchImage']:
                        search_results = obs_item['result']
                        images_path = re.findall(r"Image: (.*?), Text:", search_results)
                        text_description = re.findall(r"Text: (.*?)\nImage:", search_results)
                        text_description_last = re.findall(r"Text: (.+)$", search_results)
                        text_description_list = text_description + text_description_last
                        # images_path = images_path[:5]
                        images_path = []
                        if len(images_path)>0:
                            raw_images_list = [process_image(image, 512*28*28, 2*28*28) for image in images_path]
                            multi_modal_data.append({'image': raw_images_list})
                            image_inputs = self.processor.image_processor(raw_images_list, return_tensors='pt')
                            multi_modal_inputs.append(image_inputs)
                            image_grid_thw = image_inputs['image_grid_thw']

                            obs_str = ''.join([f"Input image:<|vision_start|>{self.processor.image_token * (image_grid_thw_item.prod() // merge_length)}<|vision_end|>\nDescription: {description}\n" for image_grid_thw_item,description  in zip(image_grid_thw, text_description_list)])
                            # raw_obs_str = f"<|vision_start|>{self.processor.image_token}<|vision_end|>" * len(image_grid_thw)
                            obs_str = '\n<|im_start|>user\n<tool_response>\nContents of retrieved images: \n' + obs_str + '</tool_response><|im_end|>\n<|im_start|>assistant\n'
                            next_obs_str.append(obs_str)
                        else:
                            # no image
                            obs_str = '\n<|im_start|>user\n<tool_response>\nContents of retrieved images: \n' + search_results + '\n</tool_response><|im_end|>\n<|im_start|>assistant\n'
                            next_obs_str.append(obs_str)
                            multi_modal_data.append({'image': []})
                            multi_modal_inputs.append(BatchFeature(dict()))
                    elif obs_item['tool'] in ['web_search','visit']:
                        results = obs_item['result']
                        obs_str = '\n<|im_start|>user\n<tool_response>\n' + results + '\n</tool_response><|im_end|>\n<|im_start|>assistant\n'
                        next_obs_str.append(obs_str)
                        multi_modal_data.append({'image': []})
                        multi_modal_inputs.append(BatchFeature(dict()))
                    else:
                        next_obs_str.append('\n<|im_start|>user\nYour previous action is invalid.\n<|im_end|>\n<|im_start|>assistant\n')
                        multi_modal_data.append({'image': []})
                        multi_modal_inputs.append(BatchFeature(dict()))
            else:
                raise ValueError('invalid observation')
        
        
        next_obs_ids = self.tokenizer(
            next_obs_str, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        return next_obs_ids, next_obs_str, multi_modal_data, multi_modal_inputs
    
    def _concat_multi_modal_data(self, rollings, next_obs_multi_modal_data:list, next_obs_multi_modal_inputs:list):
        if not 'multi_modal_inputs' in rollings.non_tensor_batch.keys():
            rollings.non_tensor_batch['multi_modal_inputs'] = np.empty(len(next_obs_multi_modal_data), dtype=object)
            for idx, item in enumerate(next_obs_multi_modal_inputs):
                rollings.non_tensor_batch['multi_modal_inputs'][idx] = item
            rollings.non_tensor_batch['multi_modal_data'] = np.array(next_obs_multi_modal_data, dtype=object)
        else:
            for idx, multi_modal_data_item in enumerate(next_obs_multi_modal_data):
                if len(multi_modal_data_item['image']) > 0:
                    # data
                    rollings.non_tensor_batch['multi_modal_data'][idx]['image'].extend(multi_modal_data_item['image'])
                    if 'pixel_values' in rollings.non_tensor_batch['multi_modal_inputs'][idx]:
                        rollings.non_tensor_batch['multi_modal_inputs'][idx]['pixel_values'] = torch.cat((rollings.non_tensor_batch['multi_modal_inputs'][idx]['pixel_values'], next_obs_multi_modal_inputs[idx]['pixel_values']),dim=0)
                        rollings.non_tensor_batch['multi_modal_inputs'][idx]['image_grid_thw'] = torch.cat((rollings.non_tensor_batch['multi_modal_inputs'][idx]['image_grid_thw'], next_obs_multi_modal_inputs[idx]['image_grid_thw']),dim=0)
                    else:
                        rollings.non_tensor_batch['multi_modal_inputs'][idx]['pixel_values'] = next_obs_multi_modal_inputs[idx]['pixel_values']
                        rollings.non_tensor_batch['multi_modal_inputs'][idx]['image_grid_thw'] = next_obs_multi_modal_inputs[idx]['image_grid_thw']
                else:
                    pass

        return rollings
        

    def _update_rolling_state(self, rollings, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding
        if next_obs_ids.shape[1] != 0:
            new_input_ids = self.tensor_fn.concatenate_with_padding([
                rollings.batch['input_ids'],
                cur_responses,
                next_obs_ids
            ])
        else:
            new_input_ids = self.tensor_fn.concatenate_with_padding([
                rollings.batch['input_ids'],
                cur_responses
            ])
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        }, rollings.non_tensor_batch)

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None and next_obs_ids.shape[1] != 0:
            responses = self.tensor_fn.concatenate_with_padding([
                right_side['responses'],
                cur_responses,
                next_obs_ids
            ], pad_to_left=False)
        else:
            responses = self.tensor_fn.concatenate_with_padding([
                right_side['responses'],
                cur_responses,
            ], pad_to_left=False)
        
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len]}


    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        padded_non_tensor_batch = {}

        padded_ids = self.tokenizer(
            ['<|im_start|>user\nHi, who are u?<|im_end|>\n<|im_start|>assistant\n'], 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']
        padded_ids = padded_ids[0]

        pad_input_ids = torch.full_like(active_batch.batch['input_ids'][0], 151643, dtype=torch.int64)
        pad_input_ids[:len(padded_ids)] = padded_ids
        pad_attention_mask = self.tensor_fn.create_attention_mask(pad_input_ids)
        pad_input_ids = pad_input_ids.unsqueeze(0)
        pad_attention_mask = pad_attention_mask.unsqueeze(0)
        pad_position_ids = self.tensor_fn.create_position_ids(pad_attention_mask)
        
        padded_batch['attention_mask'] = torch.cat([active_batch.batch['attention_mask'], pad_attention_mask.repeat(padding_size, *[1] * (len(active_batch.batch['attention_mask'].shape) - 1))], dim=0)
        padded_batch['input_ids'] = torch.cat([active_batch.batch['input_ids'], pad_input_ids.repeat(padding_size, *[1] * (len(active_batch.batch['input_ids'].shape) - 1))], dim=0)
        padded_batch['position_ids'] = torch.cat([active_batch.batch['position_ids'], pad_position_ids.repeat(padding_size, *[1] * (len(active_batch.batch['position_ids'].shape) - 1))], dim=0)
        
        # for k, v in active_batch.batch.items():
        #     # Use first sequence as padding template
        #     pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
        #     padded_batch[k] = torch.cat([v, pad_sequence], dim=0)
        
            
        # if len(active_batch.non_tensor_batch['multi_modal_inputs'].shape)>1 and active_batch.non_tensor_batch['multi_modal_inputs'].shape[1] == 0:
        #     active_batch.non_tensor_batch.pop('multi_modal_inputs')
        #     active_batch.non_tensor_batch.pop('multi_modal_data')
        # else:
        for k, v in active_batch.non_tensor_batch.items():
            pad_non_tensor_item = np.empty(padding_size, dtype=object)
            if k == 'raw_prompt_ids':
                list_ids = padded_ids.tolist()
                for idx in range(padding_size):
                    pad_non_tensor_item[idx] = list_ids
            elif k == 'multi_modal_inputs':
                for idx in range(padding_size):
                    pad_non_tensor_item[idx] = {}
            elif k == 'multi_modal_data':
                for idx in range(padding_size):
                    pad_non_tensor_item[idx] = {'image': []}
            padded_non_tensor_batch[k] = np.concatenate([v, pad_non_tensor_item])
                
        padded_active_batch = DataProto.from_dict(padded_batch, padded_non_tensor_batch)

        # padded_active_batch = DataProto.from_dict(padded_batch, active_batch.non_tensor_batch)
        
        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def _raw_prompt_ids(self, rollings):
        new_raw_prompt_ids = []
        rollings.batch['input_ids'] = rollings.batch['input_ids'].long()
        raw_next_obs_ids = [ids[mask == 1].tolist() for ids, mask in zip(np.array(rollings.batch['input_ids']),  np.array(rollings.batch['attention_mask']))]
        def replace_consecutive_elements(arr, target):
            result = []
            i = 0
            while i < len(arr):
                if arr[i] == target:
                    result.append(target)
                    while i + 1 < len(arr) and arr[i + 1] == target:
                        i += 1
                else:
                    result.append(arr[i])
                i += 1
            return result
        raw_next_obs_ids = [replace_consecutive_elements(row,151655) for row in raw_next_obs_ids]
        raw_next_obs_ids = np.array(raw_next_obs_ids, dtype=object)
        rollings.non_tensor_batch['raw_prompt_ids'] = raw_next_obs_ids
        return rollings

    def deactivate_batch(self, active_mask,rollings):
        raw_prompt_ids = rollings.non_tensor_batch['raw_prompt_ids']
        max_model_len = 22048
        curr_active_mask = torch.tensor([len(raw_prompt_ids_item) < max_model_len for raw_prompt_ids_item in raw_prompt_ids], dtype=torch.bool)
        active_mask = active_mask * curr_active_mask
        return active_mask

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""

        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        # rollings_multimodal_data = gen_batch.non_tensor_batch.get('multi_modal_inputs', None)
        # rollings_multimodal_data = gen_batch.non_tensor_batch['multi_modal_inputs']
        # rollings_multimodal_data = None
        raw_prompt_ids = rollings.non_tensor_batch['raw_prompt_ids']

        self.retrievaled_images = [[] for _ in range(gen_batch.batch['input_ids'].shape[0])]

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            rollings = self._raw_prompt_ids(rollings)

            active_mask = self.deactivate_batch(active_mask, rollings)
            if not active_mask.sum():
                break

            if 'multi_modal_inputs' in rollings.non_tensor_batch.keys():
                rollings_active = DataProto.from_dict(
                    tensors={k: v[active_mask] for k, v in rollings.batch.items()},
                    non_tensors={k: v[active_mask] for k, v in rollings.non_tensor_batch.items()}
                )
            else:
                rollings_active = DataProto.from_dict({
                    k: v[active_mask] for k, v in rollings.batch.items()
                })

            # self.processor.batch_decode(rollings_active.batch['input_ids'])
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])

            print(responses_str[0])
            
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            # Execute in environment and process observations
            next_obs, dones = self.execute_predictions(responses_str, self.tokenizer.pad_token, active_mask)
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            next_obs_ids, next_obs_str, next_obs_multi_modal_data, next_obs_multi_modal_inputs = self._process_next_obs(next_obs, rollings)
            rollings = self._concat_multi_modal_data(
                rollings,
                next_obs_multi_modal_data,
                next_obs_multi_modal_inputs
            )
            # Update states            
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            

        # final LLM rollout
        if active_mask.sum():

            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            rollings = self._raw_prompt_ids(rollings)

            active_mask = self.deactivate_batch(active_mask, rollings)

            if active_mask.sum():

                if 'multi_modal_inputs' in rollings.non_tensor_batch.keys():
                    rollings_active = DataProto.from_dict(
                        tensors={k: v[active_mask] for k, v in rollings.batch.items()},
                        non_tensors={k: v[active_mask] for k, v in rollings.non_tensor_batch.items()}
                    )
                else:
                    rollings_active = DataProto.from_dict({
                        k: v[active_mask] for k, v in rollings.batch.items()
                    })

                gen_output = self._generate_with_gpu_padding(rollings_active)

                meta_info = gen_output.meta_info
                responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
                responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

                # # Execute in environment and process observations
                _, dones = self.execute_predictions(
                    responses_str, self.tokenizer.pad_token, active_mask, do_search=False
                )

                curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
                active_mask = active_mask * curr_active_mask
                active_num_list.append(active_mask.sum().item())

                original_right_side = self._update_right_side(
                    original_right_side,
                    responses_ids,
                )
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        # =================== raw prompt ids ===================
        rollings.non_tensor_batch['raw_prompt_ids'] = raw_prompt_ids
        
        if not self.is_validation:
            rollings, original_right_side = self._add_noisy_multi_modal_data(rollings, original_right_side)
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info, rollings)
    
    def _add_noisy_multi_modal_data(self, rollings, original_right_side):
        # from ray.util import pdb
        # pdb.set_trace()
        image_padded = Image.new('RGB', (64, 64), (0, 0, 0))

        image_padded = process_image(image_padded, 256*256, 128*128)
        image_inputs = self.processor.image_processor([image_padded], return_tensors='pt')
        image_grid_thw = image_inputs['image_grid_thw']
        merge_length = self.processor.image_processor.merge_size**2
        padded_str = f"\n<|im_start|>user\n<|vision_start|>{self.processor.image_token * (image_grid_thw.prod() // merge_length)}<|vision_end|><|im_end|>"

        padded_str_list = []
        for idx, multi_modal_item in enumerate(rollings.non_tensor_batch['multi_modal_data']):
            if len(multi_modal_item['image']) == 0:
                padded_str_list.append(padded_str)
                rollings.non_tensor_batch['multi_modal_data'][idx]['image'].append(image_padded)
                rollings.non_tensor_batch['multi_modal_inputs'][idx] = image_inputs
            else:
                padded_str_list.append('')
            
        padded_ids = self.tokenizer(
            padded_str_list, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        original_right_side = self._update_right_side(
            original_right_side,
            padded_ids
        )
        return rollings, original_right_side


    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict,
                            rollings) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )

        final_output = DataProto.from_dict(final_output,rollings.non_tensor_batch)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones = [], []
        
        tool_queries = [content for action, content in zip(cur_actions, contents) if action == 'tool_call']

        # qwen-agent
        search_results = []
        if len(tool_queries) > 0:
            def tool_call(request_para):
                try:
                    request_para = json.loads(request_para)
                    # reuslt = self.qwen_agent._call_tool(request_para['name'], request_para['arguments'],img_save_path='/mnt/data/qiuchen.wqc/code/ds_verl/images_data',byte=False)
                    reuslt = self.qwen_agent._call_tool(request_para['name'], request_para['arguments'])
                    search_result =  dict(
                        status=True,
                        tool=request_para['name'],
                        tool_arguments=request_para['arguments'],
                        result=reuslt
                    )
                    return search_result
                except Exception as e:
                    return dict(status=False)

            max_workers = min(128, len(tool_queries))
            if max_workers == 1:
                # single thread
                search_results = []
                for request_para in tool_queries:
                    search_results.append(tool_call(request_para))
            else:
                # multi_thread
                with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
                    futures = {executor.submit(tool_call, query): i for i, query in enumerate(tool_queries)}
                    search_results = [None] * len(tool_queries)
                    for future in concurrent.futures.as_completed(futures):
                        index = futures[future]  # 获取原始顺序
                        result = future.result()  # 获取任务结果
                        search_results[index] = result  # 将结果放入对应位置

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                elif action == 'tool_call':
                    obs_dict = search_results.pop(0)
                    try:
                        if obs_dict['status']:
                            next_obs.append(obs_dict)
                        else:
                            raise Exception('Tool call failed')
                    except Exception as e:
                        next_obs.append('\n<|im_start|>user\nThe tool is error.\n<|im_end|>\n<|im_start|>assistant\n')
                    dones.append(0)
                else:
                    next_obs.append('\n<|im_start|>user\nYour previous action is invalid.\n<|im_end|>\n<|im_start|>assistant\n')
                    dones.append(0)
                
        assert len(search_results) == 0
        return next_obs, dones

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(tool_call|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        # results = self._batch_search(queries)['result']
        
        # return [self._passages2string(result) for result in results]
        response = requests.get(self.config.search_url, params={"queries": queries})
        response_json = response.json()
        return response_json

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
