use std::fs::File;
use std::io::{BufReader, Result, Read};
use byteorder::ByteOrder;
use memmap2::Mmap;

type RawConfigI32 = [i32; 7];

#[derive(Debug, Default)]
struct Config {
    dim: usize, // transformer dimension
    hidden_dim: usize, // for ffn layers
    n_layers: usize, // number of layers
    n_heads: usize, // number of query heads
    n_kv_heads: usize, // number of key/value heads
    vocab_size: usize, // vocabulary size
    seq_len: usize, // max. sequence length
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
            std::mem::transmute::<[u8; std::mem::size_of::<RawConfigI32>()], RawConfigI32>(mmap[..config_byte_size].try_into().unwrap())
        };

        Ok(Self {
            dim: raw_config[0] as usize,
            hidden_dim: raw_config[1] as usize,
            n_layers: raw_config[2] as usize,
            n_heads: raw_config[3] as usize,
            n_kv_heads: raw_config[4] as usize,
            vocab_size: raw_config[5] as usize,
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
    vocab: Vec<String>
}

impl Vocab {
    fn from_file(tokenizer_file_path: &str, vocab_size: usize) -> Result<Self> {
        let mut vocab = Vocab::default();
        vocab.vocab.reserve(vocab_size);
        
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
    freq_cis_real: Vec<f32>, // (seq_len, dim/2)
    freq_cis_imag: Vec<f32>, // (seq_len, dim/2)
    // (optional) classifier weights for the logits, on the last layer
    wcls: Vec<f32>
}

impl TransformerWeights {
    fn from_file(weights_file_path: &str, config: &Config) -> Result<Self> {
        // TODO: read transformer weights
        
        let weights = TransformerWeights::default();
        Ok(weights)
    }
}

fn main() -> Result<()> {
    let file_path = "weights/stories15M.bin";
    
    let config = Config::from_file(file_path)?;
    let vocab = Vocab::from_file("tokenizer.bin", config.vocab_size as usize)?;
    let transformer_weights = TransformerWeights::from_file(file_path, &config)?;

    println!("{:?}", config); // FIXME:
    // println!("{:?}", vocab); // FIXME:
    
    // TODO: set up inference state

    // TODO: forward pass
    
    Ok(())
}
