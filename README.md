# llama2-rs

Goal: run LLaMA2 locally on a CPU (and be faster than llama2.c from Andrej Karpathy :) )

TODOs:

- [X] Implement loading of the model
- [ ] Implement forward pass
- [ ] Implement generation
- [ ] Implement benchmarking
- [ ] Optimize performance (parallelization, SIMD/vectorization, fuse loops etc.)
- [ ] Support prompting and tokenization
- [ ] Quantization


Resources to dive into the different concepts of LLaMA2:
- RoPE
    - https://arxiv.org/pdf/2104.09864v4.pdf
    - https://blog.eleuther.ai/rotary-embeddings/
- LLaMA2: https://github.com/facebookresearch/llama
