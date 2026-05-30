-- Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
-- SPDX-License-Identifier: LicenseRef-RunuX-Commercial
--
-- RunuX.lean — Top-level entry point for the RunuX Formal Verification Suite
-- ===========================================================================
--
-- This is the root module of the RunuX-AI formal specification library.
-- It re-exports all sub-modules for convenient access.
--
-- Module Structure:
--   RunuxSpec.Basic              — PFC gating axioms (NormedSpace, Homeostatic Stability, Lipschitz)
--   RunuxSpec.PFCRouter          — Calibrated PFC Router (Deductive Floor, MCTS Budget)
--   RunuxSpec.DFAAlignment       — Direct Feedback Alignment & Direct Inference Transfer
--   RunuxSpec.SpeculativeDecoding — Modified Rejection Sampling & Carbon-Aware K Scaling
--
-- Build: cd spec && lake build
-- Lean Toolchain: leanprover/lean4:v4.17.0

import RunuxSpec
