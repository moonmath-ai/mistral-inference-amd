from __future__ import annotations
import os 

from pathlib import Path

from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_inference.timing import StageTiming

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

import threading

from time import perf_counter
from typing import Dict, Optional, Tuple, Union

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

    def __call__(
        self, prompt: str, max_tokens: int = 512, temperature: float = 0.0, return_timing: bool = False
    ) -> Union[Tuple[str, int], Tuple[str, int, StageTiming]]:
        timing: Optional[StageTiming] = StageTiming() if return_timing else None
        completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
        tok_start = perf_counter()
        tokens = self._tokenizer.encode_chat_completion(completion_request).tokens
        tok_end = perf_counter()
        if timing is not None:
            timing.tokenization_ms = (tok_end - tok_start) * 1000.0
        generated_tokens, _ = generate(
            encoded_prompts=[tokens],
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            eos_id=self._tokenizer.instruct_tokenizer.tokenizer.eos_id,
            timing=timing,
        )
        nof_tokens = len(generated_tokens[0])
        decoded = self._tokenizer.instruct_tokenizer.tokenizer.decode(generated_tokens[0])
        if timing is not None:
            return decoded, nof_tokens, timing
        return decoded, nof_tokens

if __name__ == "__main__":
    chat = Chat("mistral_7b_instruct_v3")
    print(chat("Explain Machine Learning to me in a nutshell."))