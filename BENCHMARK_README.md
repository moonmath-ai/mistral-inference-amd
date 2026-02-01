# Benchmarking MistralAI models
## Relevant files
- [bench.py](./bench.py) script for measuring TPS of the model for various prompts, for different serving options.
- [chat.py](./chat.py) script for directly prompting the MistralAI models, based on examples shown in [the tutorial notebook](./tutorials/getting_started.ipynb) (If the notebook is missing see the [README](./tutorials/README.md) to generate the notebooks from the jupytext script).
- [vllm_serve.sh](./vllm_serve.sh) - script for launching a vLLM serving of the model.
## Running a benchmark
`bench.py` receives a few CLI arguments for specific benchmarks:
- `--model` - model name, fitting one of the keys seen in `MODEL_FULL_NAMES`, default is the base 7B model.
- `--vllm` - use vLLM for serving and measure its performance, default is to use MistralAI's serving.
- `--multigpu` - use multiple GPUs for serving and measurement.

The benchmark runs the relevant model serving with a list of prompts seen in the start of the file. Both prompt results and metrics (TPS etc.) are stored in `outputs/<model key name>/<native or vllm>_output_<output number>_<prompt start>__<nof gpus>`, and `outputs/<model key name>/<native or vllm>_summary__<num gpus>`.

### Mistral / native benchmark
To check their implementation, only the above `bench.py` must be run. Outputs will be stored in 
> [!NOTE]
> Current implementation of `chat.py` doesn't actually utilizes multiple GPUs, launching it will only run on one device.
### vLLM benchmark
To check the vLLM serving of the model, prior to running `bench.py`, a serving of vLLM must be launched on a separate terminal. The launch is handled by `vllm_serve.sh -m <model name> -g <multi | single>`, where model name is one of the values of the `MODEL_FULL_NAMES` dictionary.
