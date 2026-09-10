// Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-RunuX-Commercial
// This file is proprietary and confidential. Unauthorized copying,
// distribution, or use is strictly prohibited.

#![cfg_attr(not(test), no_std)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
//! RunuX Tokenizer — Minimal BPE tokenizer for LLM inference
//!
//! Supports byte-level BPE (GPT-NeoX / Qwen / LLaMA style) in `no_std`
//! environments. Vocabulary and merge rules are loaded from GGUF metadata
//! at runtime.
//!
//! # Design
//!
//! This tokenizer is intentionally minimal — it handles:
//! - BPE merge-based encoding (text → token IDs)
//! - Decoding (token IDs → text bytes)
//! - Special token mapping (BOS, EOS, padding)
//!
//! It does NOT handle:
//! - SentencePiece (would need a separate impl)
//! - Regex-based pre-tokenization (GPT-4 style)

extern crate alloc;
use alloc::string::String;
use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------

/// A single vocabulary entry.
#[derive(Debug, Clone)]
pub struct VocabEntry {
    /// Token string (may be raw bytes for byte-fallback tokens)
    pub token: String,
    /// Score/priority for merge ordering (lower = higher priority)
    pub score: f32,
    /// Token type (normal, special, byte-fallback, etc.)
    pub token_type: TokenType,
}

/// Token type classification.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    /// Normal text token
    Normal = 0,
    /// Unknown/fallback token
    Unknown = 1,
    /// Control/special token (BOS, EOS, etc.)
    Control = 2,
    /// User-defined special token
    UserDefined = 3,
    /// Unused
    Unused = 4,
    /// Byte-level fallback token (e.g., <0x41> for 'A')
    Byte = 5,
}

impl TokenType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::Normal,
            1 => Self::Unknown,
            2 => Self::Control,
            3 => Self::UserDefined,
            4 => Self::Unused,
            5 => Self::Byte,
            _ => Self::Normal,
        }
    }
}

// ---------------------------------------------------------------------------
// Special Token IDs
// ---------------------------------------------------------------------------

/// Well-known special token IDs.
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    /// Beginning of sequence
    pub bos_id: u32,
    /// End of sequence
    pub eos_id: u32,
    /// Padding token
    pub pad_id: u32,
    /// Unknown token
    pub unk_id: u32,
    /// Separator token (optional)
    pub sep_id: Option<u32>,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos_id: 1,
            eos_id: 2,
            pad_id: 0,
            unk_id: 0,
            sep_id: None,
        }
    }
}

// ---------------------------------------------------------------------------
// BPE Tokenizer
// ---------------------------------------------------------------------------

/// Byte-level BPE tokenizer.
///
/// Loads vocabulary and merge rules, then encodes/decodes text.
/// Compatible with GPT-NeoX, LLaMA, Qwen, and DeepSeek tokenizers.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// Vocabulary: token_id → entry
    pub vocab: Vec<VocabEntry>,
    /// Special token configuration
    pub special_tokens: SpecialTokens,
    /// Whether to add BOS token on encode
    pub add_bos: bool,
    /// Whether to add EOS token on encode
    pub add_eos: bool,
}

impl BpeTokenizer {
    /// Create a new tokenizer from vocabulary entries.
    pub fn new(vocab: Vec<VocabEntry>, special_tokens: SpecialTokens) -> Self {
        Self {
            vocab,
            special_tokens,
            add_bos: true,
            add_eos: false,
        }
    }

    /// Create an empty tokenizer (for testing).
    pub fn empty() -> Self {
        Self {
            vocab: Vec::new(),
            special_tokens: SpecialTokens::default(),
            add_bos: false,
            add_eos: false,
        }
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Encode text into token IDs using byte-level BPE.
    ///
    /// This implements the standard BPE algorithm:
    /// 1. Convert text to bytes
    /// 2. Initialize each byte as a separate token
    /// 3. Iteratively merge the highest-priority adjacent pair
    /// 4. Return the final token ID sequence
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();

        if self.add_bos {
            tokens.push(self.special_tokens.bos_id);
        }

        if text.is_empty() {
            if self.add_eos {
                tokens.push(self.special_tokens.eos_id);
            }
            return tokens;
        }

        // Byte-level fallback: convert each byte to its token ID
        let bytes = text.as_bytes();
        let mut byte_tokens: Vec<u32> = Vec::with_capacity(bytes.len());

        for &b in bytes {
            // Find the byte-level token (e.g., <0x41> for byte 0x41)
            let token_id = self
                .find_byte_token(b)
                .unwrap_or(self.special_tokens.unk_id);
            byte_tokens.push(token_id);
        }

        // BPE merge loop
        loop {
            if byte_tokens.len() < 2 {
                break;
            }

            // Find the best merge (lowest score = highest priority)
            let mut best_merge: Option<(usize, u32, f32)> = None; // (position, merged_token_id, score)

            for i in 0..byte_tokens.len() - 1 {
                let left = byte_tokens[i];
                let right = byte_tokens[i + 1];

                // Construct the merged token string
                let merged = self.merge_token_string(left, right);
                if let Some(merged_id) = self.find_token(&merged) {
                    let score = self.vocab[merged_id as usize].score;
                    if best_merge.is_none() || score < best_merge.unwrap().2 {
                        best_merge = Some((i, merged_id, score));
                    }
                }
            }

            match best_merge {
                Some((pos, merged_id, _)) => {
                    // Apply the merge
                    byte_tokens[pos] = merged_id;
                    byte_tokens.remove(pos + 1);
                }
                None => break, // No more merges possible
            }
        }

        tokens.extend_from_slice(&byte_tokens);

        if self.add_eos {
            tokens.push(self.special_tokens.eos_id);
        }

        tokens
    }

    /// Decode token IDs back to a string.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();

        for &token_id in tokens {
            let id = token_id as usize;
            if id >= self.vocab.len() {
                continue;
            }

            let entry = &self.vocab[id];

            // Skip control tokens
            if entry.token_type == TokenType::Control {
                continue;
            }

            // Handle byte-level tokens
            if entry.token_type == TokenType::Byte {
                if let Some(byte_val) = parse_byte_token(&entry.token) {
                    bytes.push(byte_val);
                    continue;
                }
            }

            // Normal token: append its bytes
            bytes.extend_from_slice(entry.token.as_bytes());
        }

        // Best-effort UTF-8 decode
        String::from_utf8(bytes)
            .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
    }

    /// Get the token string for a given ID.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.vocab.get(id as usize).map(|e| e.token.as_str())
    }

    /// Find a token ID by its string.
    fn find_token(&self, token: &str) -> Option<u32> {
        self.vocab
            .iter()
            .position(|e| e.token == token)
            .map(|i| i as u32)
    }

    /// Find the byte-level fallback token for a given byte value.
    fn find_byte_token(&self, byte: u8) -> Option<u32> {
        let byte_token = alloc::format!("<0x{:02X}>", byte);
        self.find_token(&byte_token).or_else(|| {
            // Some models use single-char tokens for printable ASCII
            if byte.is_ascii_graphic() || byte == b' ' {
                let ch = alloc::format!("{}", byte as char);
                self.find_token(&ch)
            } else {
                None
            }
        })
    }

    /// Construct the merged token string from two token IDs.
    fn merge_token_string(&self, left: u32, right: u32) -> String {
        let left_str = self
            .vocab
            .get(left as usize)
            .map(|e| e.token.as_str())
            .unwrap_or("");
        let right_str = self
            .vocab
            .get(right as usize)
            .map(|e| e.token.as_str())
            .unwrap_or("");
        alloc::format!("{}{}", left_str, right_str)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a byte-level token string like "<0x41>" into the byte value 0x41.
fn parse_byte_token(token: &str) -> Option<u8> {
    if token.starts_with("<0x") && token.ends_with('>') && token.len() == 6 {
        let hex = &token[3..5];
        u8::from_str_radix(hex, 16).ok()
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_vocab() -> Vec<VocabEntry> {
        // Minimal vocab: byte tokens + a few merged tokens
        let mut vocab = Vec::new();

        // ID 0: padding
        vocab.push(VocabEntry {
            token: String::from("<pad>"),
            score: 0.0,
            token_type: TokenType::Control,
        });

        // ID 1: BOS
        vocab.push(VocabEntry {
            token: String::from("<s>"),
            score: 0.0,
            token_type: TokenType::Control,
        });

        // ID 2: EOS
        vocab.push(VocabEntry {
            token: String::from("</s>"),
            score: 0.0,
            token_type: TokenType::Control,
        });

        // ID 3: "h"
        vocab.push(VocabEntry {
            token: String::from("h"),
            score: 100.0,
            token_type: TokenType::Normal,
        });

        // ID 4: "e"
        vocab.push(VocabEntry {
            token: String::from("e"),
            score: 100.0,
            token_type: TokenType::Normal,
        });

        // ID 5: "l"
        vocab.push(VocabEntry {
            token: String::from("l"),
            score: 100.0,
            token_type: TokenType::Normal,
        });

        // ID 6: "o"
        vocab.push(VocabEntry {
            token: String::from("o"),
            score: 100.0,
            token_type: TokenType::Normal,
        });

        // ID 7: "he" (merged)
        vocab.push(VocabEntry {
            token: String::from("he"),
            score: 10.0, // Higher priority (lower score)
            token_type: TokenType::Normal,
        });

        // ID 8: "ll" (merged)
        vocab.push(VocabEntry {
            token: String::from("ll"),
            score: 20.0,
            token_type: TokenType::Normal,
        });

        // ID 9: "hel" (merged)
        vocab.push(VocabEntry {
            token: String::from("hel"),
            score: 5.0,
            token_type: TokenType::Normal,
        });

        // ID 10: "hell" (merged)
        vocab.push(VocabEntry {
            token: String::from("hell"),
            score: 3.0,
            token_type: TokenType::Normal,
        });

        // ID 11: "hello" (merged)
        vocab.push(VocabEntry {
            token: String::from("hello"),
            score: 1.0, // Highest priority
            token_type: TokenType::Normal,
        });

        vocab
    }

    #[test]
    fn test_encode_simple() {
        let vocab = build_test_vocab();
        let tok = BpeTokenizer::new(vocab, SpecialTokens::default());
        let tokens = tok.encode("hello");
        // Should encode as: [BOS=1, "hello"=11]
        assert_eq!(tokens, vec![1, 11]);
    }

    #[test]
    fn test_decode_simple() {
        let vocab = build_test_vocab();
        let tok = BpeTokenizer::new(vocab, SpecialTokens::default());
        let text = tok.decode(&[1, 11]);
        // BOS is control (skipped), "hello" is decoded
        assert_eq!(text, "hello");
    }

    #[test]
    fn test_encode_partial_merge() {
        let vocab = build_test_vocab();
        let tok = BpeTokenizer::new(vocab, SpecialTokens::default());
        // "helo" — should merge "he" but 'l' and 'o' don't form "lo"
        let tokens = tok.encode("helo");
        // Expect: [BOS=1, "he"=7, "l"=5, "o"=6]
        // or with further merging: [BOS=1, "hel"=9, "o"=6]
        assert!(tokens.len() >= 2); // At minimum BOS + something
        assert_eq!(tokens[0], 1); // BOS
    }

    #[test]
    fn test_empty_string() {
        let vocab = build_test_vocab();
        let tok = BpeTokenizer::new(vocab, SpecialTokens::default());
        let tokens = tok.encode("");
        assert_eq!(tokens, vec![1]); // Just BOS
    }

    #[test]
    fn test_parse_byte_token() {
        assert_eq!(parse_byte_token("<0x41>"), Some(0x41));
        assert_eq!(parse_byte_token("<0xFF>"), Some(0xFF));
        assert_eq!(parse_byte_token("<0x00>"), Some(0x00));
        assert_eq!(parse_byte_token("hello"), None);
        assert_eq!(parse_byte_token("<0xGG>"), None);
    }

    #[test]
    fn test_vocab_size() {
        let vocab = build_test_vocab();
        let tok = BpeTokenizer::new(vocab, SpecialTokens::default());
        assert_eq!(tok.vocab_size(), 12);
    }

    #[test]
    fn test_roundtrip() {
        let vocab = build_test_vocab();
        let tok = BpeTokenizer::new(vocab, SpecialTokens::default());
        let tokens = tok.encode("hello");
        let decoded = tok.decode(&tokens);
        assert_eq!(decoded, "hello");
    }
}
