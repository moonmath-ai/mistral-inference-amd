from __future__ import annotations
import os 

from pathlib import Path

from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

import threading

from typing import Dict

class ChatMeta(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs) -> Chat:
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
            elif cls._instances[cls].model_name != args[0]:
                raise ValueError("Model name provided is different from the existing singleton instance.")
            return cls._instances[cls]

class Chat(metaclass=ChatMeta):
    def __init__(self, model_path: str):
        model_dir = Path(model_path)
        tikenizer_file = next(model_dir.glob("tokenizer*"), None)
        if tikenizer_file is None:
            raise FileNotFoundError(f"Tokenizer file not found in {model_path}")
        self._tokenizer = MistralTokenizer.from_file(str(tikenizer_file))
        self._model = Transformer.from_folder(model_path)
        self._model_name = model_path
        print(f"Chat instance created for model: {model_path}")

    def __call__(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
        tokens = self._tokenizer.encode_chat_completion(completion_request).tokens
        generated_tokens, _ = generate(
            encoded_prompts=[tokens],
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            eos_id=self._tokenizer.instruct_tokenizer.tokenizer.eos_id
        )
        nof_tokens = len(generated_tokens[0])
        return self._tokenizer.instruct_tokenizer.tokenizer.decode(generated_tokens[0]), nof_tokens

if __name__ == "__main__":
    chat = Chat("mistral_7b_instruct_v3")
    print(chat("Explain Machine Learning to me in a nutshell."))