use byteorder::ByteOrder;
use memmap2::Mmap;
use std::fs::File;
use std::io::{BufReader, Read, Result};

type RawConfigI32 = [i32; 7];

#[derive(Debug, Default)]
struct Config {
    dim: usize,        // transformer dimension
    hidden_dim: usize, // for ffn layers
    n_layers: usize,   // number of layers
    n_heads: usize,    // number of query heads
    head_size: usize, // size of each head (dim / n_heads)
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

fn read_n<R>(reader: R, bytes_to_read: u64) -> Result<Vec<u8>>
where
    R: Read,
{
    let mut buf = vec![];
    let mut chunk = reader.take(bytes_to_read);
    let n = chunk.read_to_end(&mut buf)?;
    assert_eq!(bytes_to_read as usize, n);
    Ok(buf)
}

#[derive(Debug, Default)]
struct Vocab {
    vocab: Vec<String>,
}

impl Vocab {
    fn from_file(tokenizer_file_path: &str, vocab_size: usize) -> Result<Self> {
        let mut vocab = Vocab::default();
        vocab.vocab.reserve(vocab_size);

        // TODO: use mmap
        let file = File::open(tokenizer_file_path)?;
        let mut reader = BufReader::new(file);

        for _ in 0..vocab_size {
            // Read length from file stream
            let length_buffer = read_n(&mut reader, std::mem::size_of::<i32>() as u64)?;
            let string_length = byteorder::LittleEndian::read_i32(&length_buffer);

            // Read string from file stream
            let string_buffer = read_n(&mut reader, string_length as u64)?;
            let string = String::from_utf8(string_buffer).expect("could not read word");
            vocab.vocab.push(string);
        }

        Ok(vocab)
    }

    fn get_word(&self, token: usize) -> &str {
        &self.vocab[token]
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
    freq_cis_real: Vec<f32>, // (seq_len, dim/2) TODO: the comment says that this is dim/2, but the code says head_size/2 where head_size = dim / n_heads???
    freq_cis_imag: Vec<f32>, // (seq_len, dim/2)
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

        let mut offset = std::mem::size_of::<Config>();

        // Read the weights
        let token_embedding_table_size = config.vocab_size * config.dim;
        let token_embedding_table: Vec<f32> =
            byte_chunk_to_vec(&mmap[offset..], token_embedding_table_size);
        offset += token_embedding_table_size * std::mem::size_of::<f32>();

        // Read the RMSNorm weights
        let rms_att_weight_size = config.n_layers * config.dim;
        let rms_att_weight: Vec<f32> = byte_chunk_to_vec(&mmap[offset..], rms_att_weight_size);
        offset += rms_att_weight_size * std::mem::size_of::<f32>();

        let rms_ffn_weight_size = config.n_layers * config.dim;
        let rms_ffn_weight: Vec<f32> = byte_chunk_to_vec(&mmap[offset..], rms_ffn_weight_size);
        offset += rms_ffn_weight_size * std::mem::size_of::<f32>();

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
}

// TODO: I think passing the size as parameter is not necessary since slice supports len()

// F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn add_vectors(target: &mut [f32], source: &[f32], n: usize) {
    for i in 0..n {
        target[i] += source[i];
    }
}

fn rmsnorm(x: &mut [f32], weight: &[f32], n: usize) {
    let mut squared_sum = 0.0;
    for i in 0..n {
        squared_sum += x[i] * x[i];
    }
    let rms = 1. / (squared_sum / n as f32).sqrt();

    for i in 0..n {
        x[i] = x[i] * rms * weight[i];
    }
}

fn rmsnorm_with_dest(dest: &mut [f32], x: &[f32], weight: &[f32], n: usize) {
    let mut squared_sum = 0.0;
    for i in 0..n {
        squared_sum += x[i] * x[i];
    }
    let rms = 1. / (squared_sum / n as f32).sqrt();

    for i in 0..n {
        dest[i] = x[i] * rms * weight[i];
    }
}

fn softmax(logits: &mut [f32], n: usize) {
    // Find max. for fixing stability
    let mut max_logit = std::f32::MIN;
    for i in 0..n {
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
fn matmul(target: &mut [f32], w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) {
    // TODO: parallelize loop
    (0..out_dim).into_iter().for_each(|i| {
        target[i] = 0.0;
        let row_offset = i * out_dim;
        for j in 0..in_dim {
            target[i] += w[row_offset + j] * x[j];
        }
    });
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
    key_cache: Vec<f32>,   // (n_kv_heads, seq_len, dim)
    value_cache: Vec<f32>, // (n_kv_heads, seq_len, dim)
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
            self.q.as_mut_slice(), // out: (dim,)
            &self.transformer.wq[weight_from..weight_to], // W: (dim, dim)
            self.xb.as_slice(), // x: (dim,)
            self.config.dim,
            self.config.dim
        );

        matmul(
            self.k.as_mut_slice(), // out: (dim,)
            &self.transformer.wk[weight_from..weight_to], // W: (dim, dim)
            self.xb.as_slice(), // x: (dim,)
            self.config.dim,
            self.config.dim
        );

        matmul(
            self.v.as_mut_slice(), // out: (dim,)
            &self.transformer.wv[weight_from..weight_to], // W: (dim, dim)
            self.xb.as_slice(), // x: (dim,)
            self.config.dim,
            self.config.dim
        );
    }

    fn attn_rope(&mut self, layer: usize, pos: usize) {
        // apply RoPE rotation to the q and k vectors for each head

        let freq_cis_real_offset = pos * self.config.head_size / 2;
        let freq_cis_imag_offset = pos * self.config.head_size / 2;
        
        for h in 0..self.config.n_heads {
            let q = &mut self.q[h * self.config.head_size..];
            let k = &mut self.k[h * self.config.head_size..];

            // rotate q and k by the freq_cis_real and freq_cis_imag
            // For more information checkout the Roformer paper,
            // section 3.4.2: https://arxiv.org/pdf/2104.09864.pdf
            for i in (0..self.config.head_size).step_by(2) {
                let cos = self.transformer.freq_cis_real[freq_cis_real_offset + i / 2];
                let sin = self.transformer.freq_cis_imag[freq_cis_imag_offset + i / 2];
                
                // Query vector
                let q0 = q[i];
                let q1 = q[i + 1];

                q[i] = q0 * cos - q1 * sin;
                q[i + 1] = q1 * cos + q0 * sin;

                // Key vector
                let k0 = k[i];
                let k1 = k[i + 1];

                k[i] = k0 * cos - k1 * sin;
                k[i + 1] = k1 * cos + k0 * sin;
            }
        }
    }

    fn multihead_attn(&mut self, layer: usize) {
        // TODO: parallelize loop
        (0..self.config.n_heads).into_iter().for_each(|h| {
            // TODO: attention
        });
    }

    // multi-head attention with RoPE
    fn attn(&mut self, layer: usize, pos: usize) {
        // qkv matmuls
        // PyTorch: xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        self.attn_qkv_matmuls(layer);
        
        // apply RoPE rotation to the q and k vectors for each head
        self.attn_rope(layer, pos);
        
        // TODO: caching

        // Multi-head attention
        self.multihead_attn(layer);

        // Map attention scores to logits
        // PyTorch: x = self.wo(x)
        let weight_from = layer * self.config.dim * self.config.dim;
        let weight_to = (layer + 1) * self.config.dim * self.config.dim;
        matmul(
            self.xb2.as_mut_slice(), // out: (dim,)
            &self.transformer.wo[weight_from..weight_to], // W: (dim, dim)
            self.xb.as_slice(), // x: (dim,)
            self.config.dim,
            self.config.dim
        );
    }

    // PyTorch: self.w2(F.silu(self.w1(x)) * self.w3(x))
    fn ffn(&mut self, layer: usize) {
        let weight_from = layer * self.config.hidden_dim * self.config.dim;
        let weight_to = (layer + 1) * self.config.hidden_dim * self.config.dim;

        // PyTorch: self.w1(x)
        matmul(
            self.hb.as_mut_slice(), // out: (hidden_dim,)
            &self.transformer.w1[weight_from..weight_to], // W: (hidden_dim, dim)
            self.xb.as_slice(), // x: (dim,)
            self.config.hidden_dim,
            self.config.dim
        );

        // PyTorch: self.w3(x)
        matmul(
            self.hb2.as_mut_slice(), // out: (hidden_dim,)
            &self.transformer.w3[weight_from..weight_to], // W: (hidden_dim, dim)
            self.xb.as_slice(), // x: (dim,)
            self.config.hidden_dim,
            self.config.dim
        );

        // PyTorch: x = F.silu(self.w1(x)) * self.w3(x)
        // Note: Fused the activation and elementwise multiplication loop
        // TODO: benchmark in llama2.c the effect of this loop fusion
        for i in 0..self.config.hidden_dim {
            self.hb[i] = silu(self.hb[i]) * self.hb2[i];
        }

        // PyTorch: self.w2(x)
        matmul(
            self.xb.as_mut_slice(), // out: (hidden_dim,)
            &self.transformer.w2[weight_from..weight_to], // W: (hidden_dim, dim)
            self.hb.as_slice(), // x: (dim,)
            self.config.dim,
            self.config.hidden_dim
        );
    }

    fn layer(&mut self, layer: usize, pos: usize) {
        // PyTorch: h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        // Note: we leave the buffer x as it is because we need it for the residual connection
        rmsnorm_with_dest(
            self.xb.as_mut_slice(),
            self.x.as_slice(),
            &self.transformer.rms_ffn_weight[layer * self.config.dim..(layer + 1) * self.config.dim],
            self.config.dim
        );
        self.attn(layer, pos);
        // residual connection
        add_vectors(
            self.x.as_mut_slice(),
            self.xb2.as_slice(),
            self.config.dim
        );

        // PyTorch: out = h + self.feed_forward.forward(self.ffn_norm(h))
        // Note: we leave the buffer x as it is because we need it for the residual connection
        rmsnorm_with_dest(
            self.xb.as_mut_slice(),
            self.x.as_slice(),
            &self.transformer.rms_ffn_weight[layer * self.config.dim..(layer + 1) * self.config.dim],
            self.config.dim
        );
        self.ffn(layer);
        // residual connection
        add_vectors(
            self.x.as_mut_slice(),
            self.xb.as_slice(),
            self.config.dim
        );
    }

    fn forward(&mut self, token: usize, pos: usize) {
        // fetch the token embedding
        // PyTorch: h = self.tok_embeddings(tokens)
        self.x.copy_from_slice(
            &self
                .transformer
                .token_embedding_table[(token * self.config.dim)..((token + 1) * self.config.dim)]
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
            self.config.dim
        );

        // generate logits, i.e., map activations from dim to vocab_size
        // PyTorch: output = self.output(h).float()
        matmul(
            self.logits.as_mut_slice(), // out: (vocab_size,)
            self.transformer.wcls.as_slice(), // W: (vocab_size, dim)
            self.x.as_slice(), // x: (dim,)
            self.config.vocab_size,
            self.config.dim
        );
    }
}

fn main() -> Result<()> {
    let file_path = "weights/stories15M.bin";

    println!("Loading config...");
    let config = Config::from_file(file_path)?;
    println!("{:?}", config);

    println!("Loading vocab...");
    let vocab = Vocab::from_file("tokenizer.bin", config.vocab_size as usize)?;

    println!("Loading weights...");
    let transformer_weights = TransformerWeights::from_file(file_path, &config)?;

    println!("Done.");

    // TODO: set up inference state

    // TODO: forward pass

    Ok(())
}
