# ==============================================================================
# RunuX AI Runtime — Tests for Navier-Stokes Advisor & Physics Guard
#
# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# Licensed under LicenseRef-RunuX-Commercial.
# ==============================================================================

import math

import pytest
from runux.navier_stokes_advisor import (
    NavierStokesAdvisor,
    PhysicsGuard,
    TruncationSuggestion,
    PreconditionerSuggestion,
    TimeStepSuggestion,
    AdaptiveMeshSuggestion,
    ValidationResult,
    HorizonPrediction,
    ReadabilityAdvisor,
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


# ---------------------------------------------------------------------------
# Dimensional consistency of the time-step heuristic — a control that can FAIL
# ---------------------------------------------------------------------------

class TestTimeStepDimensions:
    """A physically meaningful dt must be invariant under a change of the LENGTH unit.

    Rescale lengths by λ with time fixed: ν → λ²ν, u_max → λ·u_max, and enstrophy
    Z = Σ|k|²|û_k|² ~ |∇u|² ~ [T⁻²] is unchanged.  A dt with the dimension of time
    must come back identical.  The legacy estimate ε = ν·u² is off by L², so its dt
    is NOT invariant — this test asserts that too, so it demonstrably distinguishes
    the two forms rather than merely passing on both.
    """

    NU, U, Z = 0.05, 12.0, 2.8e4
    LAMBDA = 3.0

    def _dt(self, nu, u, z=None):
        adv = NavierStokesAdvisor(cfl_limit=0.5)
        return adv.suggest_time_step(current_dt=1e-3, cfl_number=0.3, max_velocity=u,
                                     viscosity=nu, enstrophy=z).suggested_dt

    def test_enstrophy_form_is_invariant_under_length_rescaling(self):
        lam = self.LAMBDA
        dt0 = self._dt(self.NU, self.U, self.Z)
        dt1 = self._dt(lam ** 2 * self.NU, lam * self.U, self.Z)
        assert dt0 > 0.0
        assert abs(dt1 - dt0) / dt0 < 1e-12, f"dt changed under length rescaling: {dt0} -> {dt1}"

    def test_legacy_form_is_not_invariant_negative_control(self):
        lam = self.LAMBDA
        dt0 = self._dt(self.NU, self.U)                      # no enstrophy: legacy path
        dt1 = self._dt(lam ** 2 * self.NU, lam * self.U)
        # η_legacy = √(ν/u) scales as √λ, so dt_cfl scales as 1/√λ and dt_visc as 1/λ:
        # either way the legacy dt moves. If it ever stops moving, this control has
        # stopped discriminating and the invariance test above proves nothing.
        assert abs(dt1 - dt0) / dt0 > 0.1, "legacy dt unexpectedly invariant; control is dead"

    def test_legacy_path_is_flagged_and_lower_confidence(self):
        adv = NavierStokesAdvisor(cfl_limit=0.5)
        with_z = adv.suggest_time_step(1e-3, 0.3, self.U, self.NU, enstrophy=self.Z)
        without = adv.suggest_time_step(1e-3, 0.3, self.U, self.NU)
        assert "LEGACY" in without.reasoning and "LEGACY" not in with_z.reasoning
        assert without.confidence < with_z.confidence
        assert "CFL" in without.reasoning            # existing contract preserved

    def test_enstrophy_form_has_correct_kolmogorov_scale(self):
        # η = (ν³/ε)^{1/4} with ε = 2νZ, checked against a hand calculation.
        nu, z = 0.05, 2.831e4
        eta = (nu ** 3 / (2.0 * nu * z)) ** 0.25
        assert abs(eta - 0.014495) < 2e-6
        # and the suggested dt is bounded by both limits built from that η
        dt = self._dt(nu, 12.0, z)
        assert dt <= 0.5 * eta / 12.0 + 1e-15
        assert dt <= eta ** 2 / (2 * nu) + 1e-15


# ---------------------------------------------------------------------------
# ReadabilityAdvisor — accuracy, which no stability guard can see
# ---------------------------------------------------------------------------

def _pair(env, dt, horizon, n=200):
    """Build a dt / dt/2 pilot pair whose relative disagreement follows env(t)."""
    ts = [horizon * (i + 1) / n for i in range(n)]
    b = [(t, 1.0) for t in ts]                       # reference (dt/2) is 1.0 everywhere
    a = [(t, 1.0 + env(t)) for t in ts]              # dt run deviates by env(t)
    return a, b


class TestReadabilityAdvisor:
    def test_recovers_exact_power_law(self):
        # disagreement = 0.5 * t^2  ->  hits 0.02 at t = 0.2
        adv = ReadabilityAdvisor(tolerance=0.02, integrator_order=4)
        a, b = _pair(lambda t: 0.5 * t ** 2, dt=1e-3, horizon=0.1)
        p = adv.predict(a, b, current_dt=1e-3, target_horizon=1.0)
        assert abs(p.growth_exponent - 2.0) < 1e-6
        assert abs(p.readable_horizon - 0.2) < 1e-6
        assert p.target_readable is False
        # envelope at T=1 is 0.5; need factor (0.02/0.5)^(1/4) on dt
        assert abs(p.suggested_dt - 1e-3 * (0.02 / 0.5) ** 0.25) < 1e-12

    def test_target_inside_horizon_keeps_dt(self):
        adv = ReadabilityAdvisor(tolerance=0.02)
        a, b = _pair(lambda t: 0.5 * t ** 2, dt=1e-3, horizon=0.1)
        p = adv.predict(a, b, current_dt=1e-3, target_horizon=0.15)
        assert p.target_readable is True
        assert p.suggested_dt == 1e-3

    def test_saturating_growth_is_predicted_conservatively(self):
        # Error grows like t^2 early, then saturates at 0.01 (< tolerance): the run is
        # readable FOREVER, but a power law fitted on the early pilot predicts a finite
        # horizon. The advisor must err on the SAFE side: predicted <= true (= inf).
        env = lambda t: min(0.5 * t ** 2, 0.01)
        adv = ReadabilityAdvisor(tolerance=0.02)
        a, b = _pair(env, dt=1e-3, horizon=0.12)
        p = adv.predict(a, b, current_dt=1e-3, target_horizon=10.0, pilot_horizon=0.12)
        assert p.readable_horizon < 10.0, "must not over-promise on a pilot that only saw growth"
        assert p.target_readable is False   # conservative: a false 'unreadable', never a false 'readable'

    def test_raw_oscillation_is_handled_by_envelope(self):
        # Raw disagreement oscillates (dephasing); the envelope is monotone t^1.5.
        env = lambda t: 0.3 * t ** 1.5 * (0.5 + 0.5 * abs(math.sin(80 * t)))
        adv = ReadabilityAdvisor(tolerance=0.02)
        a, b = _pair(env, dt=1e-3, horizon=0.3)
        p = adv.predict(a, b, current_dt=1e-3, target_horizon=1.0)
        assert 1.2 < p.growth_exponent < 1.8
        assert p.confidence > 0.5

    def test_too_few_points_is_low_confidence_not_an_error(self):
        adv = ReadabilityAdvisor(tolerance=0.02, min_pilot_points=4)
        a = [(0.1, 1.001), (0.2, 1.002)]
        b = [(0.1, 1.0), (0.2, 1.0)]
        p = adv.predict(a, b, current_dt=1e-3, target_horizon=1.0)
        assert p.confidence == 0.0 and p.target_readable is False
        assert "longer pilot" in p.reasoning

    def test_non_growing_disagreement(self):
        adv = ReadabilityAdvisor(tolerance=0.02)
        a, b = _pair(lambda t: 1e-6, dt=1e-3, horizon=0.5)   # flat: no growth at all
        p = adv.predict(a, b, current_dt=1e-3, target_horizon=100.0)
        assert p.readable_horizon == math.inf
        assert p.target_readable is True

    def test_predict_all_is_governed_by_the_most_sensitive_observable(self):
        # Observable 1 barely moves; observable 2 dephases fast. A rule of the form
        # "both must agree" is a rule on the max, so the fast one must set the horizon.
        # Measured origin: energy alone gave a false 'readable' where enstrophy did not.
        slow_a, slow_b = _pair(lambda t: 1e-4 * t, dt=1e-3, horizon=0.1)
        fast_a, fast_b = _pair(lambda t: 0.5 * t ** 2, dt=1e-3, horizon=0.1)
        adv = ReadabilityAdvisor(tolerance=0.02)
        alone = adv.predict(slow_a, slow_b, current_dt=1e-3, target_horizon=1.0)
        both = adv.predict_all([(slow_a, slow_b), (fast_a, fast_b)],
                               current_dt=1e-3, target_horizon=1.0)
        assert alone.target_readable is True          # the slow observable alone is fooled
        assert both.target_readable is False          # the combination is not
        assert abs(both.readable_horizon - 0.2) < 1e-6

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError):
            ReadabilityAdvisor(tolerance=1.5)
        with pytest.raises(ValueError):
            ReadabilityAdvisor(integrator_order=0)


class TestStepHalvingGuard:
    def test_within_and_outside_tolerance(self):
        assert PhysicsGuard.check_step_halving(1.000, 1.010, tolerance=0.02) is True
        assert PhysicsGuard.check_step_halving(1.000, 1.050, tolerance=0.02) is False

    def test_stability_guards_cannot_see_an_accuracy_failure(self):
        # A step that is CFL-safe, energy-consistent and bounded, yet unreadable:
        # exactly the failure mode measured on the M=16 horizon-6 runs.
        assert PhysicsGuard.check_cfl_stability(dt=2.5e-4, dx=0.1, max_velocity=12.0, cfl_limit=0.5)
        assert PhysicsGuard.check_energy_conservation(2.2385e-2, 2.2385e-2 * (1 + 1e-12))
        assert PhysicsGuard.check_enstrophy_bound(6.08e-2, bound=1e6)
        # ...and the only guard that notices:
        assert PhysicsGuard.check_step_halving(2.2385e-2, 2.3600e-2, tolerance=0.02) is False
