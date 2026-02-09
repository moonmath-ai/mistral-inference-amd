Status:

(1) Mistral (at least 7B model I use now) was optimized for inference on consumer grade GPUs, and not on datacenter-grade MI300X.

(2) The best approach for datacenter-grade GPUs: increase out metric TPS (Transaction Per Second) by serving not just a single user, but many of them at the same time. This leads us to using and *optimizing* vLLM. I think Koren got vLLM working, the next step - we could try to speed it up.

(3) CUDA Graphs shouldn't help with Mistral since it uses Autoregressive decode with KV-cache (one token at a time). Examples of problems: The sequence length isn't fixed; using KV-cache is unpredictable (hit/miss).

(4) Regarding attention. Mistral uses attention from Meta's xformer. We could try FlashAttention-2 with ROCm backends or PyTorch SDPA. If we focus on multi-user processing (vLLM) we could try vLLM’s ROCm attention backend.

(5) Regarding torch compiler. Most promising is prompt processing stage (prefill) since it works with larger GEMMs. Compiling prefil would be beneficial for for longer prompts.


I believe we have to make a decision what to optimize first: single-user or multi-user (vLLM) solution.

My opinion: since MI300X is datacenter-grade we should focus on multi-user solution.