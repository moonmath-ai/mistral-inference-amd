# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jupytext_scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Getting Started with `mistral-inference`
#
# This notebook will guide you through the process of running Mistral models locally. We will cover the following: 
# - How to chat with Mistral 7B Instruct
# - How to run Mistral 7B Instruct with function calling capabilities
#
# We recommend using a GPU such as the A100 to run this notebook. 

# %% id="G6tXvIsQenpI"
# !pip install mistral-inference

# %% [markdown]
# ## Download Mistral 7B Instruct

# %% colab={"background_save": true} id="4ytmRt0WQeMW"
# !wget https://models.mistralcdn.com/mistral-7b-v0-3/mistral-7B-Instruct-v0.3.tar

# %% id="eRZg_8wvs5A6"
# !DIR=$HOME/mistral_7b_instruct_v3 && mkdir -p $DIR && tar -xf mistral-7B-Instruct-v0.3.tar -C $DIR

# %% id="7CN8gShDf65M"
# !ls ~/mistral_7b_instruct_v3

# %% [markdown]
# ## Chat with the model

# %%
import os 

from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

# load tokenizer
mistral_tokenizer = MistralTokenizer.from_file(os.path.expanduser("~")+"/mistral_7b_instruct_v3/tokenizer.model.v3")
# chat completion request
completion_request = ChatCompletionRequest(messages=[UserMessage(content="Explain Machine Learning to me in a nutshell.")])
# encode message
tokens = mistral_tokenizer.encode_chat_completion(completion_request).tokens
# load model
model = Transformer.from_folder(os.path.expanduser("~")+"/mistral_7b_instruct_v3")
# generate results
out_tokens, _ = generate([tokens], model, max_tokens=64, temperature=0.0, eos_id=mistral_tokenizer.instruct_tokenizer.tokenizer.eos_id)
# decode generated tokens
result = mistral_tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
print(result)

# %% [markdown] id="ce4woS3LkgZ9"
# ## Function calling
#
# Mistral 7B Instruct v3 also supports function calling!

# %% [markdown] id="TKfPiEwNk1kh"
# Let's start by creating a function calling example

# %% id="0PJdwvDEk3dl"
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import Function, Tool

completion_request = ChatCompletionRequest(
    tools=[
        Tool(
            function=Function(
                name="get_current_weather",
                description="Get the current weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location.",
                        },
                    },
                    "required": ["location", "format"],
                },
            )
        )
    ],
    messages=[
        UserMessage(content="What's the weather like today in Paris?"),
        ],
)

# %% [markdown] id="bG6ZeZUylpBW"
# Since we have already loaded the tokenizer and the model in the example above. We will skip these steps here. 
#
# Now we can encode the message with our tokenizer using `MistralTokenizer`. 

# %% id="Ii8q-JNClwiq"
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

tokens = mistral_tokenizer.encode_chat_completion(completion_request).tokens

# %% [markdown] id="NrueDujkmJT4"
# and run `generate` to get a response. Don't forget to pass the EOS id!

# %% id="GWJYO43rl0V8"
from mistral_inference.generate import generate

out_tokens, _ = generate([tokens], model, max_tokens=64, temperature=0.0, eos_id=mistral_tokenizer.instruct_tokenizer.tokenizer.eos_id)

# %% [markdown] id="v7baJ1msmPMv"
# Finally, we can decode the generated tokens.

# %% id="RKhryfBWmHon"
result = mistral_tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens)[0]
result
