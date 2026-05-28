// Adapted from vllm_router_rs (xds-router) tokenizer module.
// Provides tokenizer abstractions (HF, tiktoken, mock) plus streaming/stop helpers.

pub mod chat_template;
pub mod factory;
pub mod hub;
pub mod huggingface;
pub mod kimi_local;
pub mod mock;
pub mod sequence;
pub mod stop;
pub mod stream;
pub mod tiktoken;
pub mod traits;

pub use chat_template::{ChatMessage, ChatTemplateProcessor};
pub use factory::{
    create_huggingface_tokenizer_async, create_tokenizer_async, create_tokenizer_from_file,
    TokenizerType,
};
pub use huggingface::HuggingFaceTokenizer;
pub use mock::MockTokenizer;
pub use sequence::Sequence;
pub use stop::{SequenceDecoderOutput, StopSequenceConfig, StopSequenceDecoder};
pub use stream::DecodeStream;
pub use tiktoken::{TiktokenModel, TiktokenTokenizer};
pub use traits::{Decoder, Encoder, Encoding, SpecialTokens, TokenIdType, Tokenizer};
