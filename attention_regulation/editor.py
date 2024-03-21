from typing import Union

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from typing import List

from transformers import CLIPTokenizer, CLIPTextModel
from attention_regulation.edit_scheme import OptimiseAttnMapEditScheme


class StableDiffusionEditor:
    def __init__(
        self,
        pipeline_id: str,
        scheduler: str,
        cuda_id: int,
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 50,
        num_edit_steps: int = 25,
        attention_layers_to_edit: List[str] = ["all"],
        guidance_scale: float = 7.5,
    ):
        self.attention_layers_to_edit = attention_layers_to_edit
        self.pipeline = DiffusionPipeline.from_pretrained(
            pipeline_id, torch_dtype=torch.float16, variant="fp16"
        )
        if scheduler == "DPMSolverMultistepScheduler":
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
        elif scheduler == "PNDMScheduler":  # this is the default scheduler in diffusers
            pass
        else:
            raise NotImplementedError(f"Scheduler {scheduler} not supported")

        self.cuda_id = cuda_id
        self.edit_scheme_instance = None
        self.num_images_per_prompt = num_images_per_prompt
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.num_edit_steps = num_edit_steps
        self.reset_internals()
        self.prompt_embedder = PromptEmbedder(pipeline_id, f"cuda:{cuda_id}")
        self.pipeline.to(f"cuda:{cuda_id}")
        self.edit_scheme = OptimiseAttnMapEditScheme

    def get_target_idsx(self, tokenized_text: list, tokenized_target_text: list):
        """get list of contiguous target idx in tokenized text, raise error if not found"""
        target_idsx = []
        for item in tokenized_target_text[1:-1]:  # Remove start and end tokens
            try:
                target_idsx.append(tokenized_text.index(item))
            except ValueError:
                raise ValueError(f"Target phrase {tokenized_target_text} not found")
        return target_idsx

    def process_inputs(self, prompt: str, target_phrase: str):
        # get prompt encoding and target words' id
        _, tokenized_text = self.prompt_embedder(prompt, return_tokenized_text=True)
        _, tokenized_target_text = self.prompt_embedder.tokenize_phrase(
            target_phrase, padding=False
        )
        target_phrase_ids = self.get_target_idsx(tokenized_text, tokenized_target_text)
        target_phrase_ids = [
            target_phrase_ids for _ in range(self.num_images_per_prompt)
        ]
        eos_id = tokenized_text.index("<|endoftext|>")
        print("Tokenized text: ", tokenized_text)
        print(f"Target phrase ids: {target_phrase_ids[0]}")
        print(f"Target phrase: {[tokenized_text[i] for i in target_phrase_ids[0]]}")
        return target_phrase_ids, eos_id

    def reset_internals(self) -> None:
        self.internals = {}

    def initialize_edit_hook(self, prompt: str, target_phrase: str):
        self.reset_internals()
        target_phrase_ids, ens_id = self.process_inputs(prompt, target_phrase)
        self.edit_scheme_instance = self.edit_scheme(
            target_phrase_ids,
            ens_id,
            self.internals,
            self.attention_layers_to_edit,
            self.num_edit_steps,
        )
        hook_list = self.edit_scheme_instance.add_hook(self.pipeline.unet)
        return hook_list

    def get_remove_hooks_callback(
        self, hook_list: List[torch.utils.hooks.RemovableHandle]
    ):
        def remove_hooks_callback(pipeline, i, t, callback_kwargs):
            if i != self.num_edit_steps:
                return
            for hook in hook_list:
                hook.remove()
            hook_list.clear()
            print(f"Removed hooks at step {i}")
            return callback_kwargs

        return remove_hooks_callback

    def get_current_step_callback(self):
        def current_step_callback(pipeline, i, t, callback_kwargs):
            self.edit_scheme_instance.current_step += 1
            return callback_kwargs

        return current_step_callback

    @staticmethod
    def get_call_back_wrapper(callback_fns: List[callable]):
        def call_back_wrapper(pipeline, i, t, callback_kwargs):
            for callback_fn in callback_fns:
                callback_fn(pipeline, i, t, callback_kwargs)
            return callback_kwargs

        return call_back_wrapper

    def txt2img(
        self,
        prompt: str,
        target_phrase: str,
    ):
        self.reset_internals()
        # initialize hook for every prompt
        if self.edit_scheme is not None:
            hook_list = self.initialize_edit_hook(prompt, target_phrase)
        # initialize all callbacks for diffusion pipeline
        callback_fns = []
        if self.edit_scheme is not None:
            callback_fns.append(self.get_current_step_callback())
            callback_fns.append(self.get_remove_hooks_callback(hook_list))

        # Generate images from text
        images = self.pipeline(
            prompt,
            num_images_per_prompt=self.num_images_per_prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            callback_on_step_end=self.get_call_back_wrapper(callback_fns),
            callback_on_step_end_tensor_inputs=["latents"],
        ).images
        if self.edit_scheme is not None:
            for hook in hook_list:
                hook.remove()

        return images


class PromptEmbedder:
    def __init__(
        self,
        diffusion_pipeline: Union[DiffusionPipeline, str],
        torch_device: str = "cuda:0",
    ):
        if isinstance(diffusion_pipeline, str):
            self.tokenizer = CLIPTokenizer.from_pretrained(
                diffusion_pipeline, subfolder="tokenizer"
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                diffusion_pipeline, subfolder="text_encoder"
            )
        else:
            self.tokenizer = diffusion_pipeline.tokenizer
            self.text_encoder = diffusion_pipeline.text_encoder
        self.torch_device = torch_device
        self.text_encoder.to(self.torch_device)

    def __call__(
        self,
        prompt: str,
        return_tokenized_text: bool = False,
        remove_padding: bool = False,
    ):
        # get tokenized text
        text_tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        # get tokenized text
        if return_tokenized_text:
            tokenized_text = []
            for word_id in text_tokens["input_ids"][0]:
                tokenized_text.append(
                    self.tokenizer._convert_id_to_token(word_id.item()).replace(
                        "</w>", ""
                    )
                )
            if remove_padding:
                first_padding_idx = tokenized_text.index("!")
                tokenized_text = tokenized_text[:first_padding_idx]
        # get clip embedding
        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_tokens.input_ids.to(self.torch_device)
            )[0]
            text_embeddings = text_embeddings[0]
        if not return_tokenized_text:
            return text_embeddings
        else:
            return text_embeddings, tokenized_text

    def tokenize_phrase(self, prompt: str, padding: bool = True):
        # get tokenized text
        if padding:
            text_tokens = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
        else:
            text_tokens = self.tokenizer(prompt, return_tensors="pt")

        # get tokenized text
        tokenized_text = []
        for word_id in text_tokens["input_ids"][0]:
            tokenized_text.append(
                self.tokenizer._convert_id_to_token(word_id.item()).replace("</w>", "")
            )
        return text_tokens, tokenized_text
