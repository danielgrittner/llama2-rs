# llama2-rs

LLaMA2 port for Rust inspired by `llama2.c`.

TODOs:

- [X] Implement loading of the model
- [X] Implement forward pass
- [X] Implement generation
- [X] Implement tokens/sec
- [ ] Support prompting and tokenization
- [ ] Command line args
- [X] Parallelize implementation
- [ ] Optimize performance (SIMD/vectorization, fuse loops etc.)
- [ ] (Maybe) Quantization


Current Performance on my M1 Pro:

```
tokens / seconds = 145.67

<s>
 Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.
Lily wanted to play with the ball, but it was too high up in the sky. She tried to jump and reach it, but she couldn't. Then, she had an idea. She would use a stick to knock the ball down.
Lily found a stick and tried to hit the ball. But the stick was too short. She tried again and again, but she couldn't reach it. She felt sad.
Suddenly, a kind man came by and saw Lily. He asked her what was wrong. Lily told him about the ball. The man smiled and said, "I have a useful idea!" He took out a long stick and used it to knock the ball down. Lily was so happy! She thanked the man and they played together in the sunshine.
<s>
 Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big,% 
```