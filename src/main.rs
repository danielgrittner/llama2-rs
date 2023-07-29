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
    wcls: Vec<f32>,
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
        let head_size = config.dim / config.n_heads;

        let freq_cis_real_size = config.seq_len * head_size / 2;
        let freq_cis_real: Vec<f32> = byte_chunk_to_vec(&mmap[offset..], freq_cis_real_size);
        offset += freq_cis_real_size * std::mem::size_of::<f32>();

        let freq_cis_imag_size = config.seq_len * head_size / 2;
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
