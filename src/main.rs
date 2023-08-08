use byteorder::ByteOrder;
use memmap2::Mmap;
use rand::Rng;
use rayon::iter::ParallelIterator;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator};
use rayon::slice::ParallelSliceMut;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Result};
use std::iter::zip;
use std::time::Instant;

const BOS_TOKEN: usize = 1;

type RawConfigI32 = [i32; 7];

#[derive(Debug, Default)]
struct Config {
    dim: usize,        // transformer dimension
    hidden_dim: usize, // for ffn layers
    n_layers: usize,   // number of layers
    n_heads: usize,    // number of query heads
    head_size: usize,  // size of each head (dim / n_heads)
    n_kv_heads: usize, // number of key/value heads
    shared_weights: bool,
    vocab_size: usize, // vocabulary size
    seq_len: usize,    // max. sequence length
}

impl Config {
    fn from_file(weights_file_path: &str) -> Result<Self> {
        // mmap binary weights file
        let file = File::open(weights_file_path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Read config from weights file
        let config_byte_size = std::mem::size_of::<RawConfigI32>();
        if mmap.len() < config_byte_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Something is wrong with the weights file. Are you sure you are using the correct file?",
            ));
        }

        let raw_config = unsafe {
            std::mem::transmute::<[u8; std::mem::size_of::<RawConfigI32>()], RawConfigI32>(
                mmap[..config_byte_size].try_into().unwrap(),
            )
        };

        Ok(Self {
            dim: raw_config[0] as usize,
            hidden_dim: raw_config[1] as usize,
            n_layers: raw_config[2] as usize,
            n_heads: raw_config[3] as usize,
            head_size: (raw_config[0] as usize) / (raw_config[3] as usize),
            n_kv_heads: raw_config[4] as usize,
            shared_weights: raw_config[5] > 0, // weird hack from Andrej
            vocab_size: raw_config[5].abs() as usize,
            seq_len: raw_config[6] as usize,
        })
    }
}

fn read_n<R>(reader: R, bytes_to_read: usize) -> Result<Vec<u8>>
where
    R: Read,
{
    let mut buf = vec![];
    let mut chunk = reader.take(bytes_to_read as u64);
    let n = chunk.read_to_end(&mut buf)?;
    assert_eq!(bytes_to_read, n);
    Ok(buf)
}

#[derive(Debug, Default)]
struct Tokenizer {
    vocab_scores: Vec<f32>,
    vocab: Vec<String>,
    word_to_token_id: HashMap<String, usize>,
    max_token_length: usize,
}

impl Tokenizer {
    fn from_file(tokenizer_file_path: &str, vocab_size: usize) -> Result<Tokenizer> {
        let mut vocab = Tokenizer::default();
        vocab.vocab_scores.reserve(vocab_size);
        vocab.vocab.reserve(vocab_size);

        let file = File::open(tokenizer_file_path)?;
        let mut reader = BufReader::new(file);

        // Read max_token_length
        let max_token_length_buffer = read_n(&mut reader, std::mem::size_of::<u32>())?;
        vocab.max_token_length = byteorder::LittleEndian::read_u32(&max_token_length_buffer) as usize;

        for _ in 0..vocab_size {
            // Read vocab score
            let vocab_score_buffer = read_n(&mut reader, std::mem::size_of::<f32>())?;
            let score = byteorder::LittleEndian::read_f32(&vocab_score_buffer);
            vocab.vocab_scores.push(score);

            // Read length from file stream
            let length_buffer = read_n(&mut reader, std::mem::size_of::<i32>())?;
            let string_length = byteorder::LittleEndian::read_i32(&length_buffer);

            // Read string from file stream
            let string_buffer = read_n(&mut reader, string_length as usize)?;
            let string = String::from_utf8(string_buffer).expect("could not read word");
            vocab.vocab.push(string);
        }

        vocab.word_to_token_id.reserve(vocab_size);
        vocab.vocab.iter().enumerate().for_each(|(token_id, word)| {
            vocab.word_to_token_id.insert(word.to_string(), token_id);
        });

        Ok(vocab)
    }

    fn decode(&self, token_id: usize) -> &str {
        &self.vocab[token_id]
    }

    fn lookup_word(&self, word: &str) -> Option<usize> {
        match self.word_to_token_id.get(word) {
            Some(token_id) => Some(*token_id),
            None => None
        }
    }

    fn bpe_encode(&self, s: &str) -> Vec<usize> {
        let mut tokens = Vec::new();
        tokens.reserve(s.len());
        
        // encode every individual byte
        for i in 0..s.len() {
            let token_id = self.lookup_word(&s[i..i+1]).unwrap();
            tokens.push(token_id);
        }
    
        let mut str_buffer = String::with_capacity(2 * self.max_token_length);
    
        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        loop {
            let mut best_score = -1e10;
            let mut best_token_id = usize::MAX;
            let mut best_idx = usize::MAX;

            for i in 0..tokens.len() - 1 {
                // Copy the two consecutive tokens into a single string
                str_buffer.clear();
                str_buffer.push_str(&self.vocab[tokens[i]]);
                str_buffer.push_str(&self.vocab[tokens[i + 1]]);
                
                if let Some(token_id) = self.lookup_word(&str_buffer) {
                    if self.vocab_scores[token_id] > best_score {
                        best_score = self.vocab_scores[token_id];
                        best_token_id = token_id;
                        best_idx = i;
                    }
                }
            }

            if best_idx == usize::MAX {
                break;
            }

            // Merge the best pair and delete the second token
            tokens[best_idx] = best_token_id;
            tokens.remove(best_idx + 1);
        }

        tokens
    }
}

#[derive(Debug, Default)]
struct TransformerWeights {
    // Token Embedding Table
    token_embedding_table: Vec<f32>, // (vocab_size, dim)
    // Weights for RMSNorm
    rms_att_weight: Vec<f32>, // (layer, dim)
    rms_ffn_weight: Vec<f32>, // (layer, dim)
    // Weights for matmuls in attn
    wq: Vec<f32>, // (layer, dim, dim)
    wk: Vec<f32>, // (layer, dim, dim)
    wv: Vec<f32>, // (layer, dim, dim)
    wo: Vec<f32>, // (layer, dim, dim)
    // Weights for ffn
    w1: Vec<f32>, // (layer, hidden_dim, dim)
    w2: Vec<f32>, // (layer, dim, hidden_dim)
    w3: Vec<f32>, // (layer, hidden_dim, dim)
    // final RMSNorm
    rms_final_weights: Vec<f32>, // (dim)
    // freq_cis for RoPE relatively positional embeddings
    freq_cis_real: Vec<f32>, // (seq_len, head_size/2)
    freq_cis_imag: Vec<f32>, // (seq_len, head_size/2)
    // (optional) classifier weights for the logits, on the last layer
    wcls: Vec<f32>, // (vocab_size, dim)
}

fn byte_chunk_to_vec<T>(byte_chunk: &[u8], number_elements: usize) -> Vec<T>
where
    T: Clone,
{
    unsafe {
        let data = byte_chunk.as_ptr() as *const T;
        let slice_data = std::slice::from_raw_parts(data, number_elements);
        slice_data.to_vec()
    }
}

impl TransformerWeights {
    fn from_file(weights_file_path: &str, config: &Config) -> Result<Self> {
        // mmap binary weights file
        let file = File::open(weights_file_path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let mut offset = std::mem::size_of::<RawConfigI32>();

        // Read the weights
        let token_embedding_table_size = config.vocab_size * config.dim;
        let token_embedding_table: Vec<f32> =
            byte_chunk_to_vec(&mmap[offset..], token_embedding_table_size);
        offset += token_embedding_table_size * std::mem::size_of::<f32>();

        // Read the RMSNorm weights for attention
        let rms_att_weight_size = config.n_layers * config.dim;
        let rms_att_weight: Vec<f32> = byte_chunk_to_vec(&mmap[offset..], rms_att_weight_size);
        offset += rms_att_weight_size * std::mem::size_of::<f32>();

        // Read the attention weights
        let wq_size = config.n_layers * config.dim * config.dim;
        let wq: Vec<f32> = byte_chunk_to_vec(&mmap[offset..], wq_size);
        offset += wq_size * std::mem::size_of::<f32>();

        let wk_size = config.n_layers * config.dim * config.dim;
        let wk: Vec<f32> = byte_chunk_to_vec(&mmap[offset..], wk_size);
        offset += wk_size * std::mem::size_of::<f32>();

        let wv_size = config.n_layers * config.dim * config.dim;
        let wv: Vec<f32> = byte_chunk_to_vec(&mmap[offset..], wv_size);
        offset += wv_size * std::mem::size_of::<f32>();

        let wo_size = config.n_layers * config.dim * config.dim;
        let wo: Vec<f32> = byte_chunk_to_vec(&mmap[offset..], wo_size);
        offset += wo_size * std::mem::size_of::<f32>();

        // Read the RMSNorm weights for ffn
        let rms_ffn_weight_size = config.n_layers * config.dim;
        let rms_ffn_weight: Vec<f32> = byte_chunk_to_vec(&mmap[offset..], rms_ffn_weight_size);
        offset += rms_ffn_weight_size * std::mem::size_of::<f32>();

        // Read the ffn weights
        let w1_size = config.n_layers * config.hidden_dim * config.dim;
        let w1: Vec<f32> = byte_chunk_to_vec(&mmap[offset..], w1_size);
        offset += w1_size * std::mem::size_of::<f32>();

        let w2_size = config.n_layers * config.dim * config.hidden_dim;
        let w2: Vec<f32> = byte_chunk_to_vec(&mmap[offset..], w2_size);
        offset += w2_size * std::mem::size_of::<f32>();

        let w3_size = config.n_layers * config.hidden_dim * config.dim;
        let w3: Vec<f32> = byte_chunk_to_vec(&mmap[offset..], w3_size);
        offset += w3_size * std::mem::size_of::<f32>();

        // Read the final RMSNorm weights
        let rms_final_weights_size = config.dim;
        let rms_final_weights: Vec<f32> =
            byte_chunk_to_vec(&mmap[offset..], rms_final_weights_size);
        offset += rms_final_weights_size * std::mem::size_of::<f32>();

        // Read the freq_cis for RoPE relatively positional embeddings
        let freq_cis_real_size = config.seq_len * config.head_size / 2;
        let freq_cis_real: Vec<f32> = byte_chunk_to_vec(&mmap[offset..], freq_cis_real_size);
        offset += freq_cis_real_size * std::mem::size_of::<f32>();

        let freq_cis_imag_size = config.seq_len * config.head_size / 2;
        let freq_cis_imag: Vec<f32> = byte_chunk_to_vec(&mmap[offset..], freq_cis_imag_size);
        offset += freq_cis_imag_size * std::mem::size_of::<f32>();

        // Read the classifier weights
        let wcls_size = config.vocab_size * config.dim;
        let wcls: Vec<f32> = if config.shared_weights {
            token_embedding_table.clone()
        } else {
            byte_chunk_to_vec(&mmap[offset..], wcls_size)
        };

        Ok(TransformerWeights {
            token_embedding_table,
            rms_att_weight,
            rms_ffn_weight,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            w3,
            rms_final_weights,
            freq_cis_real,
            freq_cis_imag,
            wcls,
        })
    }

    // Note: does not include the token embedding table
    fn num_parameters(&self) -> usize {
        let mut n = 0;
        n += self.rms_att_weight.len();
        n += self.wq.len();
        n += self.wk.len();
        n += self.wv.len();
        n += self.wo.len();
        n += self.rms_ffn_weight.len();
        n += self.w1.len();
        n += self.w2.len();
        n += self.w3.len();
        n += self.rms_final_weights.len();
        n += self.freq_cis_real.len();
        n += self.freq_cis_imag.len();
        n += self.wcls.len();
        n
    }

    fn memory_usage_in_bytes(&self) -> usize {
        (self.num_parameters() + self.token_embedding_table.len()) * std::mem::size_of::<f32>()
    }
}

// F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn add_vectors(target: &mut [f32], source: &[f32]) {
    target
        .iter_mut()
        .zip(source.iter())
        .for_each(|(t, s)| *t += s);
}

fn rmsnorm(x: &mut [f32], weight: &[f32]) {
    let size = x.len();

    let squared_sum = x.iter().fold(0.0, |acc, x| acc + x * x);
    let rms = 1. / (squared_sum / size as f32).sqrt();

    x.iter_mut()
        .zip(weight.iter())
        .for_each(|(x, w)| *x *= rms * w);
}

fn rmsnorm_with_dest(dest: &mut [f32], x: &[f32], weight: &[f32]) {
    let size = x.len();

    let squared_sum = x.iter().fold(0.0, |acc, x| acc + x * x);
    let rms = 1. / (squared_sum / size as f32).sqrt();

    dest.iter_mut()
        .zip(x.iter())
        .zip(weight.iter())
        .for_each(|((d, x), w)| {
            *d = x * rms * w;
        });
}

fn softmax(logits: &mut [f32]) {
    let n = logits.len();

    // Find max. for fixing stability
    let mut max_logit = logits[0];
    for i in 1..n {
        max_logit = max_logit.max(logits[i]);
    }

    // Exponentiate and sum logits
    let mut sum = 0.0;
    for i in 0..n {
        logits[i] = (logits[i] - max_logit).exp();
        sum += logits[i];
    }

    // Normalize
    for i in 0..n {
        logits[i] /= sum;
    }
}

// (out_dim, in_dim) @ (d,) -> (out_dim,)
// w @ x -> target
fn matmul(target: &mut [f32], w: &[f32], x: &[f32]) {
    let in_dim = x.len();
    target.par_iter_mut().enumerate().for_each(|(i, t)| {
        let row_offset = i * in_dim;
        *t = x
            .iter()
            .zip(w[row_offset..].iter())
            .fold(0.0, |result, (x, w)| result + x * w);
    });
}

fn inner_product(x: &[f32], y: &[f32]) -> f32 {
    zip(x, y).fold(0.0, |acc, (a, b)| acc + a * b)
}

fn argmax(x: &[f32]) -> usize {
    let mut max = std::f32::MIN;
    let mut argmax = 0;
    for (i, v) in x.iter().enumerate() {
        if *v > max {
            max = *v;
            argmax = i;
        }
    }
    argmax
}

fn sample(probs: &[f32]) -> usize {
    let mut rng = rand::thread_rng();
    let mut cdf = 0.0;
    let r = rng.gen_range(0.0..1.0);
    for (i, p) in probs.iter().enumerate() {
        cdf += p;
        if cdf > r {
            return i;
        }
    }
    probs.len() - 1
}

#[derive(Debug)]
struct LLaMA2<'a> {
    // buffers for current activations
    x: Vec<f32>,      // activation at current timestep (dim,)
    xb: Vec<f32>,     // same, but inside a residual branch (dim,)
    xb2: Vec<f32>,    // additional buffer (dim,)
    hb: Vec<f32>,     // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Vec<f32>,    // buffer for hidden dimension in the ffn (hidden_dim,)
    q: Vec<f32>,      // query (dim,)
    k: Vec<f32>,      // key (dim,)
    v: Vec<f32>,      // value (dim,)
    att: Vec<f32>,    // attention scores (n_heads, seq_len)
    logits: Vec<f32>, // output logits (vocab_size,)
    // kv cache
    key_cache: Vec<f32>,   // (layer, seq_len, dim)
    value_cache: Vec<f32>, // (layer, seq_len, dim)
    // weights & config
    transformer: &'a TransformerWeights,
    config: &'a Config,
}

impl<'a> LLaMA2<'a> {
    fn new(transformer: &'a TransformerWeights, config: &'a Config) -> LLaMA2<'a> {
        Self {
            x: vec![0.0; config.dim],
            xb: vec![0.0; config.dim],
            xb2: vec![0.0; config.dim],
            hb: vec![0.0; config.hidden_dim],
            hb2: vec![0.0; config.hidden_dim],
            q: vec![0.0; config.dim],
            k: vec![0.0; config.dim],
            v: vec![0.0; config.dim],
            att: vec![0.0; config.n_heads * config.seq_len],
            logits: vec![0.0; config.vocab_size],
            key_cache: vec![0.0; config.n_kv_heads * config.seq_len * config.dim],
            value_cache: vec![0.0; config.n_kv_heads * config.seq_len * config.dim],
            transformer,
            config,
        }
    }

    // PyTorch: xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    fn attn_qkv_matmuls(&mut self, layer: usize) {
        let weight_from = layer * self.config.dim * self.config.dim;
        let weight_to = (layer + 1) * self.config.dim * self.config.dim;

        matmul(
            self.q.as_mut_slice(),                        // out: (dim,)
            &self.transformer.wq[weight_from..weight_to], // W: (dim, dim)
            self.xb.as_slice(),                           // x: (dim,)
        );

        matmul(
            self.k.as_mut_slice(),                        // out: (dim,)
            &self.transformer.wk[weight_from..weight_to], // W: (dim, dim)
            self.xb.as_slice(),                           // x: (dim,)
        );

        matmul(
            self.v.as_mut_slice(),                        // out: (dim,)
            &self.transformer.wv[weight_from..weight_to], // W: (dim, dim)
            self.xb.as_slice(),                           // x: (dim,)
        );
    }

    fn attn_rope(&mut self, layer: usize, pos: usize) {
        // apply RoPE rotation to the q and k vectors for each head

        let freq_cis_real_offset = pos * self.config.head_size / 2;
        let freq_cis_imag_offset = pos * self.config.head_size / 2;

        // rotate q and k by the freq_cis_real and freq_cis_imag
        // For more information checkout the Roformer paper,
        // section 3.4.2: https://arxiv.org/pdf/2104.09864.pdf
        for i in (0..self.config.dim).step_by(2) {
            let q0 = self.q[i];
            let q1 = self.q[i + 1];

            let k0 = self.k[i];
            let k1 = self.k[i + 1];

            let cos = self.transformer.freq_cis_real
                [freq_cis_real_offset + (i % self.config.head_size) / 2];
            let sin = self.transformer.freq_cis_imag
                [freq_cis_imag_offset + (i % self.config.head_size) / 2];

            self.q[i] = q0 * cos - q1 * sin;
            self.q[i + 1] = q1 * cos + q0 * sin;

            self.k[i] = k0 * cos - k1 * sin;
            self.k[i + 1] = k1 * cos + k0 * sin;
        }
    }

    fn cache_kv(&mut self, layer: usize, pos: usize) {
        // cache the key, value for the current timestep (pos)
        let layer_offset = layer * self.config.seq_len * self.config.dim; // offset to get to the cache of the current layer
        let cache_from = layer_offset + pos * self.config.dim;
        let cache_to = layer_offset + (pos + 1) * self.config.dim;

        self.key_cache[cache_from..cache_to].copy_from_slice(&self.k.as_slice());
        self.value_cache[cache_from..cache_to].copy_from_slice(&self.v.as_slice());
    }

    fn multihead_attn(&mut self, layer: usize, pos: usize) {
        let layer_offset_for_cache = layer * self.config.seq_len * self.config.dim; // offset to get to the cache of the current layer

        let sqrt_d = (self.config.head_size as f32).sqrt();

        self.att
            .par_chunks_exact_mut(self.config.seq_len)
            .zip(self.xb.par_chunks_exact_mut(self.config.head_size))
            .enumerate()
            .for_each(|(h, (attn_scores, xb))| {
                assert_eq!(attn_scores.len(), self.config.seq_len);
                assert_eq!(xb.len(), self.config.head_size);

                // get query vector of the timestep pos for the current head
                let q_from = h * self.config.head_size;
                let q_to = (h + 1) * self.config.head_size;
                let q = &self.q[q_from..q_to];

                // Compute temp = (K * q_pos) / sqrt(dim)
                for t in 0..=pos {
                    let timestep_and_layer_offset = layer_offset_for_cache + t * self.config.dim; // key_cache[l, t]
                                                                                                  // for the current key, we need to select the correct range which corresponds to the current head
                    let key_vector_from = timestep_and_layer_offset + h * self.config.head_size;
                    let key_vector_to = timestep_and_layer_offset + (h + 1) * self.config.head_size;
                    let key_vector = &self.key_cache[key_vector_from..key_vector_to];

                    attn_scores[t] = inner_product(q, key_vector) / sqrt_d;
                }

                // softmax the scores to get attention weights, from 0..pos inclusively
                // Compute temp2 = softmax(temp)
                softmax(&mut attn_scores[..(pos + 1)]);

                // Compute temp2^T * V
                xb.fill(0.0);

                for t in 0..=pos {
                    let timestep_and_layer_offset = layer_offset_for_cache + t * self.config.dim; // value_cache[l, t]
                                                                                                  // for the current value, we need to select the correct range which corresponds to the current head
                    let value_vector_from = timestep_and_layer_offset + h * self.config.head_size;
                    let value_vector_to =
                        timestep_and_layer_offset + (h + 1) * self.config.head_size;
                    let value_vector = &self.value_cache[value_vector_from..value_vector_to];

                    // weighted sum with attention scores as weights
                    let attention_weight = attn_scores[t];
                    for i in 0..self.config.head_size {
                        xb[i] += attention_weight * value_vector[i];
                    }
                }
            });
    }

    // multi-head attention with RoPE
    fn attn(&mut self, layer: usize, pos: usize) {
        // qkv matmuls
        // PyTorch: xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        self.attn_qkv_matmuls(layer);

        // apply RoPE rotation to the q and k vectors for each head
        self.attn_rope(layer, pos);

        // Multi-head attention with caching
        // Idea:
        //
        // Let the current sequence length until the current timestep pos be n.
        // The idea is to only compute the attention score for the token at timestep pos.
        // Therefore, we compute for each head:
        //
        //              attn_pos = softmax((K * q_pos) / sqrt(dim))^T * V
        //
        //  where
        //      - attn_pos: attention score for timestep pos. dim(head_size,)
        //      - q_pos: query vector for timestep pos. dim(head_size,)
        //      - K/V: key/value vectors for all timesteps up to n. dim(n,head_size)
        //          ==> this is also the reason why we need the caching, to store all the previous key/value vectors
        self.cache_kv(layer, pos);
        self.multihead_attn(layer, pos);

        // Map attention scores to logits
        // PyTorch: x = self.wo(x)
        let weight_from = layer * self.config.dim * self.config.dim;
        let weight_to = (layer + 1) * self.config.dim * self.config.dim;
        matmul(
            self.xb2.as_mut_slice(),                      // out: (dim,)
            &self.transformer.wo[weight_from..weight_to], // W: (dim, dim)
            self.xb.as_slice(),                           // x: (dim,)
        );
    }

    // PyTorch: self.w2(F.silu(self.w1(x)) * self.w3(x))
    fn ffn(&mut self, layer: usize) {
        let weight_from = layer * self.config.hidden_dim * self.config.dim;
        let weight_to = (layer + 1) * self.config.hidden_dim * self.config.dim;

        // PyTorch: self.w1(x)
        matmul(
            self.hb.as_mut_slice(),                       // out: (hidden_dim,)
            &self.transformer.w1[weight_from..weight_to], // W: (hidden_dim, dim)
            self.xb.as_slice(),                           // x: (dim,)
        );

        // PyTorch: self.w3(x)
        matmul(
            self.hb2.as_mut_slice(),                      // out: (hidden_dim,)
            &self.transformer.w3[weight_from..weight_to], // W: (hidden_dim, dim)
            self.xb.as_slice(),                           // x: (dim,)
        );

        // PyTorch: x = F.silu(self.w1(x)) * self.w3(x)
        // Note: Fused the activation and elementwise multiplication loop
        for i in 0..self.config.hidden_dim {
            self.hb[i] = silu(self.hb[i]) * self.hb2[i];
        }

        // PyTorch: self.w2(x)
        matmul(
            self.xb.as_mut_slice(),                       // out: (hidden_dim,)
            &self.transformer.w2[weight_from..weight_to], // W: (hidden_dim, dim)
            self.hb.as_slice(),                           // x: (dim,)
        );
    }

    fn layer(&mut self, layer: usize, pos: usize) {
        // PyTorch: h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        // Note: we leave the buffer x as it is because we need it for the residual connection
        rmsnorm_with_dest(
            self.xb.as_mut_slice(),
            self.x.as_slice(),
            &self.transformer.rms_att_weight
                [layer * self.config.dim..(layer + 1) * self.config.dim],
        );
        self.attn(layer, pos);
        // residual connection
        add_vectors(self.x.as_mut_slice(), self.xb2.as_slice());

        // PyTorch: out = h + self.feed_forward.forward(self.ffn_norm(h))
        // Note: we leave the buffer x as it is because we need it for the residual connection
        rmsnorm_with_dest(
            self.xb.as_mut_slice(),
            self.x.as_slice(),
            &self.transformer.rms_ffn_weight
                [layer * self.config.dim..(layer + 1) * self.config.dim],
        );
        self.ffn(layer);
        // residual connection
        add_vectors(self.x.as_mut_slice(), self.xb.as_slice());
    }

    fn forward(&mut self, token: usize, pos: usize) {
        // fetch the token embedding
        // PyTorch: h = self.tok_embeddings(tokens)
        self.x.copy_from_slice(
            &self.transformer.token_embedding_table
                [(token * self.config.dim)..((token + 1) * self.config.dim)],
        );

        // Note: here it always holds that seqlen == 1 in comparison to the PyTorch implementation

        // forward through the layers
        // PyTorch:
        // for layer in self.layers:
        //     h = layer(h, start_pos, freqs_cis, mask)
        for l in 0..self.config.n_layers {
            self.layer(l, pos);
        }

        // final RMSNorm
        // PyTorch: h = self.norm(h)
        rmsnorm(
            self.x.as_mut_slice(),
            self.transformer.rms_final_weights.as_slice(),
        );

        // generate logits, i.e., map activations from dim to vocab_size
        // PyTorch: output = self.output(h).float()
        matmul(
            self.logits.as_mut_slice(),       // out: (vocab_size,)
            self.transformer.wcls.as_slice(), // W: (vocab_size, dim)
            self.x.as_slice(),                // x: (dim,)
        );
    }

    fn generate(&mut self, prompt_tokens: &Vec<usize>, n_tokens: usize, temperature: f32) -> Vec<usize> {
        let mut tokens = vec![];
        tokens.reserve(n_tokens);

        let mut token = BOS_TOKEN;
        tokens.push(token);

        // forward through the prompt to fill up the KV-cache!
        for (pos, prompt_token) in prompt_tokens.iter().enumerate() {
            self.forward(token, pos);
            token = *prompt_token;
            tokens.push(token);
        }

        // complete the prompt
        for pos in prompt_tokens.len()..(n_tokens - 1) {
            self.forward(token, pos);

            if temperature == 0.0 {
                token = argmax(self.logits.as_slice());
            } else {
                // Apply temperature and then sample.
                // If temperature < 1.0 then the distribution becomes more peaked ==> lower variance in sampling
                // If temperature > 1.0 then the distribution becomes more flat ==> higher variance in sampling
                self.logits.iter_mut().for_each(|p| *p = *p / temperature);
                softmax(&mut self.logits.as_mut_slice());
                token = sample(self.logits.as_slice());
            }

            tokens.push(token);
        }

        tokens
    }

    fn memory_usage_in_bytes(&self) -> usize {
        let mut memory_usage = 0;

        memory_usage += self.x.capacity() * std::mem::size_of::<f32>();
        memory_usage += self.xb.capacity() * std::mem::size_of::<f32>();
        memory_usage += self.xb2.capacity() * std::mem::size_of::<f32>();
        memory_usage += self.hb.capacity() * std::mem::size_of::<f32>();
        memory_usage += self.hb2.capacity() * std::mem::size_of::<f32>();
        memory_usage += self.q.capacity() * std::mem::size_of::<f32>();
        memory_usage += self.k.capacity() * std::mem::size_of::<f32>();
        memory_usage += self.v.capacity() * std::mem::size_of::<f32>();
        memory_usage += self.att.capacity() * std::mem::size_of::<f32>();
        memory_usage += self.logits.capacity() * std::mem::size_of::<f32>();
        memory_usage += self.key_cache.capacity() * std::mem::size_of::<f32>();
        memory_usage += self.value_cache.capacity() * std::mem::size_of::<f32>();

        memory_usage += self.transformer.memory_usage_in_bytes();

        memory_usage
    }
}

fn main() -> Result<()> {
    let file_path = "weights/stories15M.bin";
    let prompt = "One day, Lily met a bear";
    let temperature = 0.0;
    let steps = 256;

    // Setup

    println!("Loading config...");
    let config = Config::from_file(file_path)?;
    println!("Loaded config: {:?}", config);

    println!("Loading vocab...");
    let tokenizer = Tokenizer::from_file("tokenizer.bin", config.vocab_size as usize)?;

    println!("Loading weights...");
    let transformer_weights = TransformerWeights::from_file(file_path, &config)?;

    println!("Done.");

    println!(
        "Number of parameters: {}",
        transformer_weights.num_parameters()
    );

    // Configure rayon

    let cpus = num_cpus::get();
    let active_cpus = (cpus).max(1).min(config.n_heads);
    println!("Using {} threads", active_cpus);
    rayon::ThreadPoolBuilder::new()
        .num_threads(active_cpus)
        .build_global()
        .unwrap();

    // Inference

    let start = Instant::now();
    let mut llama2 = LLaMA2::new(&transformer_weights, &config);

    let llama_memory_mib = llama2.memory_usage_in_bytes() as f32 / ((1 as usize) << 20) as f32;
    println!("Memory usage in MiB: {llama_memory_mib}");

    let prompt_tokens = tokenizer.bpe_encode(&prompt);
    let generated_tokens = llama2.generate(&prompt_tokens, steps, temperature);

    let time_elapsed = start.elapsed().as_secs_f32();
    let tokens_per_sec = (steps as f32) / time_elapsed;
    println!("tokens / seconds = {:.2?}", tokens_per_sec);

    print!("{}", prompt);
    for token in generated_tokens {
        if token == 1 && tokenizer.decode(token).starts_with(' ') {
            print!("{}", &tokenizer.decode(token)[1..]);
        } else {
            print!("{}", tokenizer.decode(token));
        };
    }

    Ok(())
}
