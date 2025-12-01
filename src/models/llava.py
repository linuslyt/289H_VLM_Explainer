from typing import Any, Callable, Dict, Union, List

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from .image_text_model import ImageTextModel

__all__ = ["LLaVA"]


class LLaVA(ImageTextModel):

    def set_model(
        self,
    ) -> None:

        self.model_ = LlavaForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=self.local_files_only,
        )

    def get_language_model(
        self,
    ) -> Callable:

        return self.model_.language_model

    def get_lm_head(
        self,
    ) -> Callable:

        return self.model_.language_model.lm_head

    def set_processor(
        self,
    ) -> None:

        self.processor_ = AutoProcessor.from_pretrained(
            self.processor_name, local_files_only=self.local_files_only
        )
        self.tokenizer_ = self.processor_.tokenizer

    def set_preprocessor(
        self,
    ) -> None:

        self.preprocessor_ = self.preprocess_input

    def get_conversation_round(
        self, instruction: str = "What are these?", response: str = ""
    ) -> List[Dict[str, Any]]:

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image"},
                ],
            },
        ]
        if response:
            conversation.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": response},
                    ],
                },
            )

        return conversation

    # TODO: batch
    def preprocess_text(
        self,
        instruction: Union[str, List[str]] = "What are these?",
        response: Union[str, List[str]] = "",
        generation_mode: bool = False,
        **kwargs: Any,
    ) -> str:
        if isinstance(instruction, list) ^ isinstance(response, list):
            raise ValueError("instruction and response must both be lists OR both be single instances")
        is_batch = isinstance(instruction, list) and isinstance(response, list)

        if is_batch:
            conversations = [self.get_conversation_round(instruction=i, response=r) for i, r in zip(instruction, response)]
            # HuggingFace: 'We advise users to use padding_side="left" when computing batched generation as it leads to more accurate results.'
            self.tokenizer_.padding_side = "left"
        else:
            conversations = self.get_conversation_round(
                instruction=instruction, response=response
            )

        prompt = self.processor_.apply_chat_template(
            conversations, add_generation_prompt=generation_mode,
            padding=is_batch
        )
        return prompt

