// Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
#![deny(clippy::all)]
#![warn(clippy::pedantic)]

use serde::{Deserialize, Serialize};

pub const CURRENT_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct IntervalSerde {
    pub lo: f64,
    pub hi: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofCertificate {
    pub schema_version: u32,
    pub truncation_m: usize,
    pub viscosity_rational: String,
    pub time_start: IntervalSerde,
    pub time_end: IntervalSerde,
    pub h1_norm_bound: IntervalSerde,
    pub enstrophy_bound: IntervalSerde,
    pub mode_decay_alpha: IntervalSerde,
    pub mode_decay_coefficients_lo: Vec<f64>,
    pub mode_decay_coefficients_hi: Vec<f64>,
    pub invariant_region_lo: Vec<f64>,
    pub invariant_region_hi: Vec<f64>,
    pub residual_bound: IntervalSerde,
    pub contraction_rate: IntervalSerde,
    pub n_time_steps: u64,
    pub computed_at_utc: String,
    pub runux_version: String,
    pub cpu_cores_used: u32,
    pub initial_data_hash: String,
}

impl ProofCertificate {
    /// Serializes the certificate to a JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserializes a certificate from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Writes the certificate to a file.
    pub fn to_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        let json = self.to_json().map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    /// Reads a certificate from a file.
    pub fn from_file(path: &std::path::Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Self::from_json(&json).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Validates the internal consistency of the certificate fields.
    pub fn validate_consistency(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if self.schema_version != CURRENT_SCHEMA_VERSION {
            errors.push(format!("Unsupported schema version: {}", self.schema_version));
        }

        let check_interval = |int: &IntervalSerde, name: &str, errs: &mut Vec<String>| {
            if int.lo > int.hi {
                errs.push(format!("Invalid interval for {}: lo ({}) > hi ({})", name, int.lo, int.hi));
            }
        };

        check_interval(&self.time_start, "time_start", &mut errors);
        check_interval(&self.time_end, "time_end", &mut errors);
        check_interval(&self.h1_norm_bound, "h1_norm_bound", &mut errors);
        check_interval(&self.enstrophy_bound, "enstrophy_bound", &mut errors);
        check_interval(&self.mode_decay_alpha, "mode_decay_alpha", &mut errors);
        check_interval(&self.residual_bound, "residual_bound", &mut errors);
        check_interval(&self.contraction_rate, "contraction_rate", &mut errors);

        if self.mode_decay_coefficients_lo.len() != self.mode_decay_coefficients_hi.len() {
            errors.push("mode_decay_coefficients_lo and _hi lengths do not match".to_string());
        } else {
            for (i, (lo, hi)) in self.mode_decay_coefficients_lo.iter().zip(self.mode_decay_coefficients_hi.iter()).enumerate() {
                if lo > hi {
                    errors.push(format!("Invalid interval in mode_decay_coefficients at index {}: lo > hi", i));
                }
            }
        }

        if self.invariant_region_lo.len() != self.invariant_region_hi.len() {
            errors.push("invariant_region_lo and _hi lengths do not match".to_string());
        } else {
            for (i, (lo, hi)) in self.invariant_region_lo.iter().zip(self.invariant_region_hi.iter()).enumerate() {
                if lo > hi {
                    errors.push(format!("Invalid interval in invariant_region at index {}: lo > hi", i));
                }
            }
        }

        if !self.viscosity_rational.contains('/') {
            errors.push("viscosity_rational must be in 'p/q' format".to_string());
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_cert() -> ProofCertificate {
        ProofCertificate {
            schema_version: CURRENT_SCHEMA_VERSION,
            truncation_m: 10,
            viscosity_rational: "1/1000".to_string(),
            time_start: IntervalSerde { lo: 0.0, hi: 0.0 },
            time_end: IntervalSerde { lo: 1.0, hi: 1.0 },
            h1_norm_bound: IntervalSerde { lo: 0.0, hi: 10.0 },
            enstrophy_bound: IntervalSerde { lo: 0.0, hi: 5.0 },
            mode_decay_alpha: IntervalSerde { lo: 2.0, hi: 2.1 },
            mode_decay_coefficients_lo: vec![0.1, 0.2],
            mode_decay_coefficients_hi: vec![0.15, 0.25],
            invariant_region_lo: vec![-1.0, -1.0],
            invariant_region_hi: vec![1.0, 1.0],
            residual_bound: IntervalSerde { lo: 0.0, hi: 1e-10 },
            contraction_rate: IntervalSerde { lo: 0.5, hi: 0.9 },
            n_time_steps: 1000,
            computed_at_utc: "2026-09-12T00:00:00Z".to_string(),
            runux_version: "1.0.0".to_string(),
            cpu_cores_used: 64,
            initial_data_hash: "abcdef123456".to_string(),
        }
    }

    #[test]
    fn test_roundtrip() {
        let cert = dummy_cert();
        let json = cert.to_json().unwrap();
        let cert2 = ProofCertificate::from_json(&json).unwrap();
        assert_eq!(cert.schema_version, cert2.schema_version);
        assert_eq!(cert.viscosity_rational, cert2.viscosity_rational);
        assert_eq!(cert.n_time_steps, cert2.n_time_steps);
    }

    #[test]
    fn test_validation() {
        let mut cert = dummy_cert();
        assert!(cert.validate_consistency().is_ok());

        cert.time_start.lo = 2.0;
        cert.time_start.hi = 1.0;
        let errs = cert.validate_consistency().unwrap_err();
        assert!(!errs.is_empty());
    }

    #[test]
    fn test_schema_version_check() {
        let mut cert = dummy_cert();
        cert.schema_version = 999;
        let errs = cert.validate_consistency().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("Unsupported schema version")));
    }

    #[test]
    fn test_viscosity_format_check() {
        let mut cert = dummy_cert();
        cert.viscosity_rational = "0.001".to_string(); // missing '/'
        let errs = cert.validate_consistency().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("viscosity_rational must be in 'p/q' format")));
    }

    #[test]
    fn test_vector_length_mismatch() {
        let mut cert = dummy_cert();
        cert.mode_decay_coefficients_lo.push(0.5);
        let errs = cert.validate_consistency().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("mode_decay_coefficients_lo and _hi lengths do not match")));

        let mut cert2 = dummy_cert();
        cert2.invariant_region_lo.push(0.5);
        let errs2 = cert2.validate_consistency().unwrap_err();
        assert!(errs2.iter().any(|e| e.contains("invariant_region_lo and _hi lengths do not match")));
    }

    #[test]
    fn test_inverted_vector_intervals() {
        let mut cert = dummy_cert();
        cert.invariant_region_lo = vec![2.0, 1.0];
        cert.invariant_region_hi = vec![1.0, 2.0]; // first element lo > hi
        let errs = cert.validate_consistency().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("Invalid interval in invariant_region at index 0")));
    }

    #[test]
    fn test_file_io_roundtrip() {
        let cert = dummy_cert();
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join("test_proof_certificate_airgapped.json");

        cert.to_file(&temp_path).expect("Failed to write certificate file");
        let loaded = ProofCertificate::from_file(&temp_path).expect("Failed to read certificate file");

        assert_eq!(cert.schema_version, loaded.schema_version);
        assert_eq!(cert.viscosity_rational, loaded.viscosity_rational);
        assert_eq!(cert.initial_data_hash, loaded.initial_data_hash);

        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn test_air_gapped_json_content() {
        let cert = dummy_cert();
        let json = cert.to_json().unwrap();
        // Check air-gapped certificate strings for Lean 4 ingestion
        assert!(json.contains("\"truncation_m\": 10"));
        assert!(json.contains("\"viscosity_rational\": \"1/1000\""));
        assert!(json.contains("\"h1_norm_bound\""));
        assert!(json.contains("\"enstrophy_bound\""));
        assert!(json.contains("\"contraction_rate\""));
    }
}
