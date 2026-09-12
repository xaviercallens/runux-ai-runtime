# ==============================================================================
# RunuX AI Runtime — Tests for Navier-Stokes Advisor & Physics Guard
#
# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# Licensed under LicenseRef-RunuX-Commercial.
# ==============================================================================

import pytest
from runux.navier_stokes_advisor import (
    NavierStokesAdvisor,
    PhysicsGuard,
    TruncationSuggestion,
    PreconditionerSuggestion,
    TimeStepSuggestion,
    AdaptiveMeshSuggestion,
    ValidationResult,
)


class TestDataclasses:
    def test_truncation_suggestion_immutability(self):
        s = TruncationSuggestion(suggested_m=16, confidence=0.9, reasoning="Good resolution")
        assert s.suggested_m == 16
        assert s.confidence == 0.9
        assert s.reasoning == "Good resolution"
        with pytest.raises(Exception):
            s.suggested_m = 32  # frozen dataclass

    def test_preconditioner_suggestion(self):
        s = PreconditionerSuggestion(matrix_hint={"type": "diagonal"}, confidence=0.85, reasoning="Viscous")
        assert s.matrix_hint["type"] == "diagonal"
        assert s.confidence == 0.85

    def test_time_step_suggestion(self):
        s = TimeStepSuggestion(suggested_dt=0.005, confidence=0.8, reasoning="CFL bounded")
        assert s.suggested_dt == 0.005
        assert s.confidence == 0.8

    def test_adaptive_mesh_suggestion(self):
        s = AdaptiveMeshSuggestion(refinement_regions=[{"mode_start": 4, "mode_end": 8}], confidence=0.75, reasoning="Peak error")
        assert len(s.refinement_regions) == 1
        assert s.confidence == 0.75

    def test_validation_result_defaults(self):
        res = ValidationResult(accepted=True, reason="OK")
        assert res.accepted is True
        assert res.reason == "OK"
        assert res.residual_norm is None
        assert res.fallback_used is False


class TestNavierStokesAdvisor:
    def test_initialization_and_reset(self):
        advisor = NavierStokesAdvisor(min_confidence=0.75, cfl_limit=0.4, max_residual_ratio=1.05)
        assert advisor.min_confidence == 0.75
        assert advisor.cfl_limit == 0.4
        assert advisor.max_residual_ratio == 1.05
        assert advisor.acceptance_rate == 0.0

        # Simulate some validation calls
        advisor.validate_suggestion("test", None, actual_residual=1.0, previous_residual=1.0)
        advisor.validate_suggestion("test", None, actual_residual=2.0, previous_residual=1.0)
        assert advisor.acceptance_rate == 0.5

        advisor.reset_statistics()
        assert advisor.acceptance_rate == 0.0
        assert advisor._suggestions_made == 0

    def test_suggest_truncation_empty_or_zero(self):
        advisor = NavierStokesAdvisor()
        s1 = advisor.suggest_truncation_m(current_m=10, energy_spectrum=[], enstrophy=1.0)
        assert s1.confidence == 0.0
        assert s1.suggested_m == 10

        s2 = advisor.suggest_truncation_m(current_m=10, energy_spectrum=[0.0, 0.0, 0.0], enstrophy=0.0)
        assert s2.confidence == 0.0
        assert s2.suggested_m == 10

    def test_suggest_truncation_fast_decay(self):
        advisor = NavierStokesAdvisor()
        # Spectrum decays steeply: 1.0, 1e-3, 1e-6, 1e-12, 1e-15, 1e-18 ... (20 modes)
        spectrum = [1.0 * (1e-2 ** i) for i in range(20)]
        s = advisor.suggest_truncation_m(current_m=20, energy_spectrum=spectrum, enstrophy=0.5)
        # Tail energy < 1e-10 of total -> suggest decreasing M
        assert s.suggested_m < 20
        assert s.confidence > 0.8
        assert "safe to reduce M" in s.reasoning

    def test_suggest_truncation_under_resolved(self):
        advisor = NavierStokesAdvisor()
        # Flat or energetic spectrum at the tail: last mode carries substantial energy
        spectrum = [1.0] * 20
        s = advisor.suggest_truncation_m(current_m=20, energy_spectrum=spectrum, enstrophy=5.0)
        # Last mode fraction = 1/20 = 0.05 > 1e-6 -> suggest increasing M
        assert s.suggested_m == int(20 * 1.5)
        assert "under-resolved" in s.reasoning

    def test_suggest_truncation_well_resolved(self):
        advisor = NavierStokesAdvisor()
        # Tail fraction is between 1e-10 and 1e-6 relative to total
        spectrum = [1000.0] + [1e-4] * 9
        s = advisor.suggest_truncation_m(current_m=10, energy_spectrum=spectrum, enstrophy=1.0)
        # Should keep current M
        assert s.suggested_m == 10
        assert s.confidence == 0.95

    def test_suggest_preconditioner_invalid(self):
        advisor = NavierStokesAdvisor()
        s1 = advisor.suggest_preconditioner([], viscosity=0.01)
        assert s1.confidence == 0.0
        assert s1.matrix_hint == {}

        s2 = advisor.suggest_preconditioner([1.0, 2.0], viscosity=-0.01)
        assert s2.confidence == 0.0

    def test_suggest_preconditioner_valid(self):
        advisor = NavierStokesAdvisor()
        diag = [10.0, 5.0, 2.0, 1.0]
        viscosity = 0.001
        s = advisor.suggest_preconditioner(diag, viscosity=viscosity)
        assert s.confidence >= 0.5
        assert s.matrix_hint["type"] == "diagonal"
        assert len(s.matrix_hint["entries"]) == 4
        assert s.matrix_hint["viscosity"] == viscosity

    def test_suggest_time_step_invalid(self):
        advisor = NavierStokesAdvisor()
        s = advisor.suggest_time_step(current_dt=0.01, cfl_number=0.3, max_velocity=0.0, viscosity=0.01)
        assert s.confidence == 0.0
        assert s.suggested_dt == 0.01

    def test_suggest_time_step_valid(self):
        advisor = NavierStokesAdvisor(cfl_limit=0.5)
        # Normal flow parameters
        s = advisor.suggest_time_step(current_dt=0.01, cfl_number=0.9, max_velocity=1.5, viscosity=0.01)
        assert s.suggested_dt > 0.0
        assert s.confidence > 0.0
        assert "CFL" in s.reasoning

    def test_suggest_mesh_adaptation(self):
        advisor = NavierStokesAdvisor()
        spectrum = [1.0] * 10
        errors = [0.001, 0.001, 0.5, 0.8, 0.6, 0.001, 0.001, 0.001, 0.001, 0.001]
        s = advisor.suggest_mesh_adaptation(spectrum, errors)
        assert len(s.refinement_regions) >= 1
        region = s.refinement_regions[0]
        assert region["mode_start"] <= 2
        assert region["mode_end"] >= 4

    def test_validate_suggestion_acceptance_and_rejection(self):
        advisor = NavierStokesAdvisor(max_residual_ratio=1.1)

        # Residual decreased: 1.0 -> 0.8 (ratio 0.8 <= 1.1)
        res_ok = advisor.validate_suggestion("precond", None, actual_residual=0.8, previous_residual=1.0)
        assert res_ok.accepted is True
        assert res_ok.fallback_used is False

        # Residual increased beyond tolerance: 1.0 -> 1.5 (ratio 1.5 > 1.1)
        res_bad = advisor.validate_suggestion("dt", None, actual_residual=1.5, previous_residual=1.0)
        assert res_bad.accepted is False
        assert res_bad.fallback_used is True
        assert "falling back to deterministic solver" in res_bad.reason


class TestPhysicsGuard:
    def test_energy_conservation(self):
        # Within tolerance
        assert PhysicsGuard.check_energy_conservation(100.0, 100.0 + 1e-12, tolerance=1e-10) is True
        # Exceeds tolerance
        assert PhysicsGuard.check_energy_conservation(100.0, 101.0, tolerance=1e-10) is False

    def test_divergence_free(self):
        # Incompressible field (|div| = 0)
        assert PhysicsGuard.check_divergence_free(1e-15, tolerance=1e-12) is True
        # Compressible / unphysical leak
        assert PhysicsGuard.check_divergence_free(1e-5, tolerance=1e-12) is False

    def test_cfl_stability(self):
        # dt = 0.001, dx = 0.01, u = 1.0 -> CFL = 1.0 * 0.001 / 0.01 = 0.1 <= 0.5
        assert PhysicsGuard.check_cfl_stability(dt=0.001, dx=0.01, max_velocity=1.0, cfl_limit=0.5) is True
        # dt = 0.01, dx = 0.01, u = 2.0 -> CFL = 2.0 * 0.01 / 0.01 = 2.0 > 0.5
        assert PhysicsGuard.check_cfl_stability(dt=0.01, dx=0.01, max_velocity=2.0, cfl_limit=0.5) is False

    def test_enstrophy_bound(self):
        assert PhysicsGuard.check_enstrophy_bound(50.0, bound=100.0) is True
        assert PhysicsGuard.check_enstrophy_bound(150.0, bound=100.0) is False


class TestStrictPhysicalFallbackIntegration:
    def test_untrusted_ai_with_strict_fallback(self):
        """Verify the architectural invariant:
        AI acts as an untrusted optimizer. Any violation of physical bounds
        causes instant rejection and fallback to exact deterministic solver.
        """
        advisor = NavierStokesAdvisor(max_residual_ratio=1.1, cfl_limit=0.5)

        # 1. AI proposes an aggressive time step
        suggestion = advisor.suggest_time_step(
            current_dt=0.001,
            cfl_number=0.2,
            max_velocity=10.0,
            viscosity=0.001,
        )
        assert suggestion.suggested_dt > 0.0

        # 2. Physics Guard verifies CFL condition
        dx = 0.005
        is_cfl_safe = PhysicsGuard.check_cfl_stability(
            dt=suggestion.suggested_dt,
            dx=dx,
            max_velocity=10.0,
            cfl_limit=advisor.cfl_limit,
        )

        # 3. If physics guard flags instability, trigger fallback
        if not is_cfl_safe:
            fallback_res = ValidationResult(
                accepted=False,
                reason="CFL limit violated by AI time step",
                residual_norm=None,
                fallback_used=True,
            )
            assert fallback_res.fallback_used is True
            assert fallback_res.accepted is False

        # 4. If applied and residual explodes, validator activates fallback
        val_result = advisor.validate_suggestion(
            suggestion_type="time_step",
            suggestion=suggestion,
            actual_residual=50.0,   # residual exploded
            previous_residual=1.0,
        )
        assert val_result.accepted is False
        assert val_result.fallback_used is True
        assert advisor.acceptance_rate < 1.0
