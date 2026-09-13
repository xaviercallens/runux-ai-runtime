# ==============================================================================
# RunuX AI Runtime — Navier-Stokes Advisor (Untrusted AI Heuristics)
#
# Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
# Licensed under LicenseRef-RunuX-Commercial.
#
# Patent Pending: INPI Dossier Demande Provisoire de Brevet RunuX (2026).
# ==============================================================================

"""Navier-Stokes Advisor — Untrusted AI Heuristics for Spectral Solver.

This module wraps the SymBrain PFC Router to provide physics-aware
heuristic suggestions for the Navier-Stokes spectral Galerkin solver.

ARCHITECTURAL INVARIANT:
    Every AI prediction MUST be validated by the Rust computational
    kernel before use. If validation fails, the prediction is discarded
    and the deterministic exact solver is used instead.
    AI accelerates the search; it does NOT write the physics.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "TruncationSuggestion",
    "PreconditionerSuggestion",
    "TimeStepSuggestion",
    "AdaptiveMeshSuggestion",
    "ValidationResult",
    "NavierStokesAdvisor",
    "PhysicsGuard",
    "HorizonPrediction",
    "ReadabilityAdvisor",
]


# ---------------------------------------------------------------------------
# Suggestion dataclasses — each represents a SINGLE untrusted AI hint
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TruncationSuggestion:
    """Suggested spectral truncation order ``M`` from the AI heuristic.

    Attributes:
        suggested_m:  Proposed number of retained Fourier/Chebyshev modes.
        confidence:   AI self-reported confidence in [0, 1].
        reasoning:    Human-readable explanation of the suggestion.
    """
    suggested_m: int
    confidence: float
    reasoning: str


@dataclass(frozen=True)
class PreconditionerSuggestion:
    """Suggested preconditioner structure for the Newton–Krylov inner solve.

    Attributes:
        matrix_hint:  Dictionary describing diagonal/block-diagonal entries.
        confidence:   AI self-reported confidence in [0, 1].
        reasoning:    Human-readable explanation of the suggestion.
    """
    matrix_hint: dict
    confidence: float
    reasoning: str


@dataclass(frozen=True)
class TimeStepSuggestion:
    """Suggested adaptive time-step ``dt`` for the temporal integrator.

    Attributes:
        suggested_dt:  Proposed time step.
        confidence:    AI self-reported confidence in [0, 1].
        reasoning:     Human-readable explanation of the suggestion.
    """
    suggested_dt: float
    confidence: float
    reasoning: str


@dataclass(frozen=True)
class AdaptiveMeshSuggestion:
    """Suggested spectral mesh refinement regions.

    Attributes:
        refinement_regions:  List of dicts, each with ``mode_start``,
                             ``mode_end``, and ``reason`` keys.
        confidence:          AI self-reported confidence in [0, 1].
        reasoning:           Human-readable explanation of the suggestion.
    """
    refinement_regions: list[dict]
    confidence: float
    reasoning: str


@dataclass
class ValidationResult:
    """Outcome of the Rust kernel validating an AI suggestion.

    Attributes:
        accepted:       ``True`` if the suggestion passed validation.
        reason:         Human-readable explanation of the decision.
        residual_norm:  Post-application residual norm (``None`` if not computed).
        fallback_used:  ``True`` if the deterministic solver was used instead.
    """
    accepted: bool
    reason: str
    residual_norm: float | None = None
    fallback_used: bool = False


# ---------------------------------------------------------------------------
# NavierStokesAdvisor — main SymBrain integration surface
# ---------------------------------------------------------------------------

class NavierStokesAdvisor:
    """Untrusted AI advisor for the spectral Galerkin Navier-Stokes solver.

    All suggestions produced by this class are *heuristic guesses*.  The Rust
    computational kernel MUST validate every suggestion against the governing
    PDEs before applying it.  On validation failure, the kernel silently falls
    back to the deterministic exact solver.

    Parameters:
        min_confidence:      Minimum AI confidence required to emit a suggestion.
        cfl_limit:           Maximum CFL number allowed for time-step suggestions.
        max_residual_ratio:  Maximum ratio ``r_new / r_old`` accepted by the
                             generic validator (values > 1 indicate divergence).
    """

    def __init__(
        self,
        min_confidence: float = 0.8,
        cfl_limit: float = 0.5,
        max_residual_ratio: float = 1.1,
    ):
        self.min_confidence = min_confidence
        self.cfl_limit = cfl_limit
        self.max_residual_ratio = max_residual_ratio

        # --- internal accounting ---
        self._suggestions_made: int = 0
        self._suggestions_accepted: int = 0
        self._suggestions_rejected: int = 0

    # -- statistics --------------------------------------------------------

    @property
    def acceptance_rate(self) -> float:
        """Fraction of suggestions accepted by the Rust kernel."""
        if self._suggestions_made == 0:
            return 0.0
        return self._suggestions_accepted / self._suggestions_made

    def reset_statistics(self) -> None:
        """Zero all internal suggestion counters."""
        self._suggestions_made = 0
        self._suggestions_accepted = 0
        self._suggestions_rejected = 0

    # -- truncation --------------------------------------------------------

    def suggest_truncation_m(
        self,
        current_m: int,
        energy_spectrum: list[float],
        enstrophy: float,
    ) -> TruncationSuggestion:
        """Suggest whether the spectral truncation order *M* should change.

        The heuristic inspects the tail of the energy spectrum:

        * If the last 10 % of modes carry less than ``1e-10`` of the total
          energy, *M* can be safely reduced.
        * If the last mode carries more than ``1e-6`` of the total energy,
          *M* should be increased to resolve active scales.

        Args:
            current_m:       Current number of retained modes.
            energy_spectrum:  Energy at each wavenumber (length ≥ 1).
            enstrophy:       Total enstrophy of the flow field.

        Returns:
            A :class:`TruncationSuggestion` with the proposed *M*.
        """
        if not energy_spectrum:
            return TruncationSuggestion(
                suggested_m=current_m,
                confidence=0.0,
                reasoning="Empty energy spectrum — no suggestion possible.",
            )

        total_energy = sum(energy_spectrum)
        if total_energy <= 0.0:
            return TruncationSuggestion(
                suggested_m=current_m,
                confidence=0.0,
                reasoning="Total energy is zero — spectrum uninformative.",
            )

        n_modes = len(energy_spectrum)
        tail_start = max(1, n_modes - n_modes // 10)
        tail_energy = sum(energy_spectrum[tail_start:])
        tail_fraction = tail_energy / total_energy
        last_fraction = energy_spectrum[-1] / total_energy

        # Case 1: tail is negligible → reduce M
        if tail_fraction < 1e-10:
            new_m = max(4, int(current_m * 0.8))
            confidence = min(1.0, 0.9 - math.log10(max(tail_fraction, 1e-300)) / 30.0)
            return TruncationSuggestion(
                suggested_m=new_m,
                confidence=confidence,
                reasoning=(
                    f"Tail modes ({tail_start}–{n_modes - 1}) carry "
                    f"{tail_fraction:.2e} of total energy; safe to reduce M "
                    f"from {current_m} to {new_m}."
                ),
            )

        # Case 2: last mode is still energetic → increase M
        if last_fraction > 1e-6:
            new_m = int(current_m * 1.5)
            confidence = min(1.0, 0.85 + last_fraction * 1e4)
            return TruncationSuggestion(
                suggested_m=new_m,
                confidence=confidence,
                reasoning=(
                    f"Last mode carries {last_fraction:.2e} of total energy; "
                    f"spectrum is under-resolved — suggest increasing M from "
                    f"{current_m} to {new_m}."
                ),
            )

        # Default: keep current M
        return TruncationSuggestion(
            suggested_m=current_m,
            confidence=0.95,
            reasoning="Energy spectrum is well-resolved at current M.",
        )

    # -- preconditioner ----------------------------------------------------

    def suggest_preconditioner(
        self,
        jacobian_diagonal: list[float],
        viscosity: float,
    ) -> PreconditionerSuggestion:
        """Suggest a diagonal preconditioner for the Newton–Krylov solver.

        Constructs a diagonal preconditioner by scaling each mode's Jacobian
        entry by the viscous decay rate ``ν k²``, which dominates the
        high-wavenumber spectrum of the linearised Navier-Stokes operator.

        Args:
            jacobian_diagonal:  Diagonal of the Jacobian at each wavenumber.
            viscosity:          Kinematic viscosity ``ν``.

        Returns:
            A :class:`PreconditionerSuggestion` containing the hint dict.
        """
        if not jacobian_diagonal or viscosity <= 0.0:
            return PreconditionerSuggestion(
                matrix_hint={},
                confidence=0.0,
                reasoning="Insufficient data for preconditioner suggestion.",
            )

        diag_entries: list[float] = []
        for k, jac_k in enumerate(jacobian_diagonal):
            viscous_scale = viscosity * (k + 1) ** 2
            entry = 1.0 / max(abs(jac_k) + viscous_scale, 1e-15)
            diag_entries.append(entry)

        # Condition number estimate for confidence scoring
        diag_max = max(diag_entries)
        diag_min = min(diag_entries)
        cond_est = diag_max / max(diag_min, 1e-15)
        confidence = max(0.5, min(1.0, 1.0 - math.log10(max(cond_est, 1.0)) / 20.0))

        return PreconditionerSuggestion(
            matrix_hint={
                "type": "diagonal",
                "entries": diag_entries,
                "viscosity": viscosity,
                "condition_estimate": cond_est,
            },
            confidence=confidence,
            reasoning=(
                f"Diagonal preconditioner with {len(diag_entries)} entries "
                f"derived from viscous decay rates (ν={viscosity:.2e}, "
                f"estimated condition ≈ {cond_est:.1e})."
            ),
        )

    # -- time step ---------------------------------------------------------

    def suggest_time_step(
        self,
        current_dt: float,
        cfl_number: float,
        max_velocity: float,
        viscosity: float,
        enstrophy: float | None = None,
    ) -> TimeStepSuggestion:
        """Suggest an adaptive time step honouring the CFL stability limit.

        The proposed ``dt`` is the largest step satisfying:

        .. math::
            \\text{CFL} = u_{\\max} \\, \\Delta t / \\Delta x \\le \\text{cfl\\_limit}

        where ``Δx`` is estimated from the viscous Kolmogorov scale
        ``η = (ν³ / ε)^{1/4}``.

        **Dimensional note.** ``η`` is a length only if ``ε`` is the dissipation
        rate per unit mass, ``[L² T⁻³]``.  That is ``ε = 2 ν Z`` with ``Z`` the
        enstrophy ``Σ |k|² |û_k|²`` ``[T⁻²]`` — a quantity every spectral solver
        already computes.  The earlier form ``ε ≈ ν · u_max²`` is ``[L⁴ T⁻³]``,
        off by ``L²``, so its ``η`` collapsed to ``√(ν / u_max)`` with dimension
        ``L^{1/2}``: not a length, and the ``dt`` built on it was not a CFL step
        (it changed under a pure rescaling of the length unit).  Pass
        ``enstrophy`` to get the dimensionally consistent estimate; without it the
        legacy form is used at reduced confidence, so callers can see which one
        they got.

        **Scope note.** CFL and the viscous limit are *stability* constraints.
        They say nothing about *accuracy*: a run can be perfectly stable and still
        accumulate enough truncation error to be unreadable.  For that, use
        :class:`ReadabilityAdvisor`.

        Args:
            current_dt:    Current time step.
            cfl_number:    Current CFL number of the simulation.
            max_velocity:  Maximum velocity magnitude in the domain.
            viscosity:     Kinematic viscosity ``ν``.
            enstrophy:     Total enstrophy ``Z = Σ|k|²|û_k|²`` (recommended).

        Returns:
            A :class:`TimeStepSuggestion` with the proposed ``dt``.
        """
        if max_velocity <= 0.0 or viscosity <= 0.0:
            return TimeStepSuggestion(
                suggested_dt=current_dt,
                confidence=0.0,
                reasoning="Non-positive velocity or viscosity — cannot suggest dt.",
            )

        # Kolmogorov length scale as proxy for Δx
        if enstrophy is not None and enstrophy > 0.0:
            epsilon = 2.0 * viscosity * enstrophy          # [L² T⁻³]: dimensionally consistent
            dimensional_note = ""
            confidence_scale = 1.0
        else:
            epsilon = viscosity * max_velocity ** 2         # legacy: [L⁴ T⁻³], NOT a true ε
            dimensional_note = (
                " (LEGACY estimate without enstrophy: η is not a length; "
                "pass enstrophy for a dimensionally consistent dt)"
            )
            confidence_scale = 0.5
        eta = (viscosity ** 3 / max(epsilon, 1e-30)) ** 0.25
        dx_estimate = max(eta, 1e-15)

        dt_cfl = self.cfl_limit * dx_estimate / max_velocity

        # Viscous stability constraint: dt < dx² / (2ν)
        dt_viscous = dx_estimate ** 2 / (2.0 * viscosity)
        dt_suggested = min(dt_cfl, dt_viscous)

        # Confidence based on how far the current CFL is from the limit
        cfl_ratio = cfl_number / max(self.cfl_limit, 1e-15)
        if cfl_ratio > 1.0:
            confidence = max(0.5, 1.0 - (cfl_ratio - 1.0))
            reasoning = (
                f"CFL number {cfl_number:.3f} exceeds limit {self.cfl_limit}; "
                f"strongly recommend reducing dt from {current_dt:.3e} to "
                f"{dt_suggested:.3e}."
            )
        elif cfl_ratio > 0.8:
            confidence = 0.85
            reasoning = (
                f"CFL number {cfl_number:.3f} is near the limit; modest "
                f"reduction suggested (dt {current_dt:.3e} → {dt_suggested:.3e})."
            )
        else:
            confidence = 0.7
            reasoning = (
                f"CFL number {cfl_number:.3f} is well within bounds; current "
                f"dt is acceptable but {dt_suggested:.3e} is the CFL-optimal value."
            )

        return TimeStepSuggestion(
            suggested_dt=dt_suggested,
            confidence=confidence * confidence_scale,
            reasoning=reasoning + dimensional_note,
        )

    # -- mesh adaptation ---------------------------------------------------

    def suggest_mesh_adaptation(
        self,
        energy_spectrum: list[float],
        mode_errors: list[float],
    ) -> AdaptiveMeshSuggestion:
        """Suggest spectral regions that need additional resolution.

        Scans ``mode_errors`` for contiguous blocks exceeding a relative
        threshold and recommends refinement there.

        Args:
            energy_spectrum:  Energy at each wavenumber.
            mode_errors:      Estimated error at each wavenumber.

        Returns:
            An :class:`AdaptiveMeshSuggestion` listing refinement regions.
        """
        if not mode_errors or not energy_spectrum:
            return AdaptiveMeshSuggestion(
                refinement_regions=[],
                confidence=0.0,
                reasoning="Empty spectrum or error data — no suggestion possible.",
            )

        n_modes = len(mode_errors)
        max_error = max(mode_errors)
        if max_error <= 0.0:
            return AdaptiveMeshSuggestion(
                refinement_regions=[],
                confidence=0.95,
                reasoning="All mode errors are zero; mesh appears adequate.",
            )

        threshold = 0.1 * max_error
        regions: list[dict] = []
        region_start: int | None = None

        for k in range(n_modes):
            if mode_errors[k] > threshold:
                if region_start is None:
                    region_start = k
            else:
                if region_start is not None:
                    peak_err = max(mode_errors[region_start:k])
                    regions.append({
                        "mode_start": region_start,
                        "mode_end": k - 1,
                        "peak_error": peak_err,
                        "reason": (
                            f"Modes {region_start}–{k - 1} have error "
                            f"up to {peak_err:.2e} (>{threshold:.2e} threshold)."
                        ),
                    })
                    region_start = None

        # Close any trailing region
        if region_start is not None:
            peak_err = max(mode_errors[region_start:])
            regions.append({
                "mode_start": region_start,
                "mode_end": n_modes - 1,
                "peak_error": peak_err,
                "reason": (
                    f"Modes {region_start}–{n_modes - 1} have error "
                    f"up to {peak_err:.2e} (>{threshold:.2e} threshold)."
                ),
            })

        confidence = min(1.0, 0.8 + 0.05 * len(regions)) if regions else 0.5
        return AdaptiveMeshSuggestion(
            refinement_regions=regions,
            confidence=confidence,
            reasoning=(
                f"Identified {len(regions)} region(s) requiring spectral "
                f"refinement (error threshold = {threshold:.2e})."
            ),
        )

    # -- generic validation ------------------------------------------------

    def validate_suggestion(
        self,
        suggestion_type: str,
        suggestion: Any,
        actual_residual: float,
        previous_residual: float,
    ) -> ValidationResult:
        """Validate an AI suggestion by comparing residual norms.

        The generic rule is simple: if the residual norm increased by more
        than ``max_residual_ratio``, the suggestion is rejected and the
        deterministic solver is used.

        Statistics counters are updated on every call.

        Args:
            suggestion_type:    Human-readable label (e.g. ``"truncation"``).
            suggestion:         The suggestion object that was applied.
            actual_residual:    Residual norm *after* applying the suggestion.
            previous_residual:  Residual norm *before* the suggestion.

        Returns:
            A :class:`ValidationResult` recording the decision.
        """
        self._suggestions_made += 1

        if previous_residual <= 0.0:
            # Cannot compute ratio; accept if residual is finite
            accepted = math.isfinite(actual_residual)
            self._suggestions_accepted += int(accepted)
            self._suggestions_rejected += int(not accepted)
            return ValidationResult(
                accepted=accepted,
                reason=(
                    "Previous residual is zero; accepted by finiteness check."
                    if accepted
                    else "Non-finite residual — suggestion rejected."
                ),
                residual_norm=actual_residual,
                fallback_used=not accepted,
            )

        ratio = actual_residual / previous_residual
        accepted = ratio <= self.max_residual_ratio

        if accepted:
            self._suggestions_accepted += 1
            reason = (
                f"[{suggestion_type}] Accepted: residual ratio "
                f"{ratio:.4f} ≤ {self.max_residual_ratio}."
            )
            logger.info(reason)
        else:
            self._suggestions_rejected += 1
            reason = (
                f"[{suggestion_type}] Rejected: residual ratio "
                f"{ratio:.4f} > {self.max_residual_ratio}; "
                f"falling back to deterministic solver."
            )
            logger.warning(reason)

        return ValidationResult(
            accepted=accepted,
            reason=reason,
            residual_norm=actual_residual,
            fallback_used=not accepted,
        )


# ---------------------------------------------------------------------------
# PhysicsGuard — conservation-law sanity checks
# ---------------------------------------------------------------------------

class PhysicsGuard:
    """Lightweight conservation-law guard for AI suggestions.

    Every check returns ``True`` when the physical invariant is satisfied
    and ``False`` when it is violated.  The Rust kernel calls these checks
    before committing any AI-suggested parameter change.
    """

    @staticmethod
    def check_energy_conservation(
        before: float,
        after: float,
        tolerance: float = 1e-10,
    ) -> bool:
        """Verify that total kinetic energy is conserved within *tolerance*.

        Args:
            before:     Total energy before applying the suggestion.
            after:      Total energy after applying the suggestion.
            tolerance:  Maximum acceptable relative deviation.

        Returns:
            ``True`` if ``|E_after − E_before| / max(|E_before|, 1)`` ≤ tolerance.
        """
        ref = max(abs(before), 1.0)
        deviation = abs(after - before) / ref
        if deviation > tolerance:
            logger.warning(
                "Energy conservation violated: ΔE/E = %.2e (tol = %.2e)",
                deviation,
                tolerance,
            )
            return False
        return True

    @staticmethod
    def check_divergence_free(
        field_divergence: float,
        tolerance: float = 1e-12,
    ) -> bool:
        """Verify the velocity field is divergence-free (incompressibility).

        Args:
            field_divergence:  L² norm of ∇·u over the domain.
            tolerance:         Maximum acceptable divergence norm.

        Returns:
            ``True`` if ``|∇·u|`` ≤ tolerance.
        """
        if abs(field_divergence) > tolerance:
            logger.warning(
                "Divergence-free violated: |∇·u| = %.2e (tol = %.2e)",
                abs(field_divergence),
                tolerance,
            )
            return False
        return True

    @staticmethod
    def check_cfl_stability(
        dt: float,
        dx: float,
        max_velocity: float,
        cfl_limit: float,
    ) -> bool:
        """Verify the CFL condition for explicit time integration.

        Args:
            dt:            Time step.
            dx:            Grid spacing (or minimum spectral wavelength).
            max_velocity:  Maximum velocity magnitude in the domain.
            cfl_limit:     Upper bound on the CFL number.

        Returns:
            ``True`` if ``u_max · dt / dx`` ≤ cfl_limit.
        """
        if dx <= 0.0:
            logger.warning("Non-positive grid spacing dx = %.2e", dx)
            return False
        cfl = max_velocity * dt / dx
        if cfl > cfl_limit:
            logger.warning(
                "CFL stability violated: CFL = %.4f (limit = %.4f)",
                cfl,
                cfl_limit,
            )
            return False
        return True

    @staticmethod
    def check_enstrophy_bound(
        enstrophy: float,
        bound: float,
    ) -> bool:
        """Verify enstrophy remains below a prescribed upper bound.

        In 2-D turbulence enstrophy is bounded; in 3-D the bound is
        user-specified and acts as a blow-up sentinel.

        Args:
            enstrophy:  Current enstrophy value.
            bound:      Maximum allowable enstrophy.

        Returns:
            ``True`` if ``enstrophy`` ≤ ``bound``.
        """
        if enstrophy > bound:
            logger.warning(
                "Enstrophy bound exceeded: Ω = %.4e (bound = %.4e)",
                enstrophy,
                bound,
            )
            return False
        return True

    @staticmethod
    def check_step_halving(
        value_dt: float,
        value_half_dt: float,
        tolerance: float = 0.02,
    ) -> bool:
        """The readability criterion: a number computed at step ``dt`` is trusted
        only where its ``dt/2`` partner agrees within *tolerance*.

        This is the one guard in this class that tests *accuracy* rather than
        *stability* or a conservation law.  It is integrator- and PDE-agnostic,
        and it is the check that catches the failure mode the others cannot see:
        a run that stays bounded, conserves what it should, satisfies CFL, and is
        nonetheless wrong because truncation error has accumulated.

        Args:
            value_dt:       Observable at time ``t`` from the run with step ``dt``.
            value_half_dt:  The same observable at the same ``t`` from step ``dt/2``.
            tolerance:      Maximum relative disagreement.

        Returns:
            ``True`` if ``|a − b| / max(|b|, tiny)`` ≤ tolerance.
        """
        ref = max(abs(value_half_dt), 1e-300)
        gap = abs(value_dt - value_half_dt) / ref
        if gap > tolerance:
            logger.warning(
                "Step-halving disagreement %.3e exceeds tolerance %.3e: not readable",
                gap,
                tolerance,
            )
            return False
        return True


# ---------------------------------------------------------------------------
# ReadabilityAdvisor — predict how far a run can be trusted, BEFORE running it
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HorizonPrediction:
    """Predicted readable horizon of a step-halving pair, from a short pilot.

    Attributes:
        readable_horizon:   Predicted time at which the pair's disagreement
                            first exceeds the tolerance (``inf`` if the fitted
                            growth is non-positive).
        growth_exponent:    Fitted ``p`` in ``disagreement ≈ C · t^p``.
        target_readable:    Whether the requested target horizon is predicted to
                            be readable at the current step.
        suggested_dt:       Step that is predicted to make the target readable
                            (equals the current step when it already is).
        pilot_horizon:      How much of the trajectory the prediction was fitted
                            on.  A prediction is only as good as its pilot; see
                            ``reasoning``.
        confidence:         Self-reported confidence in [0, 1].
        reasoning:          Human-readable explanation.
    """
    readable_horizon: float
    growth_exponent: float
    target_readable: bool
    suggested_dt: float
    pilot_horizon: float
    confidence: float
    reasoning: str


class ReadabilityAdvisor:
    """Untrusted heuristic: from a short pilot of a ``dt`` / ``dt/2`` pair, predict
    the horizon over which the pair will satisfy the step-halving criterion, and
    the step needed to reach a target horizon.

    Integrator- and PDE-agnostic: it needs only two time series of any scalar
    observables sampled at shared times.  It exists because the guards in
    :class:`PhysicsGuard` other than :meth:`PhysicsGuard.check_step_halving` are
    stability and conservation checks, and a run can pass all of them while
    accumulating enough truncation error to be worthless.  Measured example that
    motivated it: six-hour horizon-6 runs at ``M = 16`` that were bounded,
    smooth, CFL-safe, and readable only to ``t ≈ 0.85``.  A pilot covering the
    first ``0.2`` of the horizon would have predicted that.

    Method.  The relative disagreement between the two members of the pair
    oscillates as they dephase, so the fit is to its **running maximum**
    (envelope) rather than to the raw signal, as ``C · t^p`` by log–log least
    squares.  The predicted readable horizon is where the envelope reaches the
    tolerance.  For an integrator of order ``q`` the disagreement scales as
    ``dt^q``, so the step that brings the envelope at the target horizon down to
    the tolerance is ``dt · (tol / envelope(T))^{1/q}``.

    **Use every observable your reading rule uses**, via :meth:`predict_all`.
    Validated on six ``M = 16`` pairs whose true readable horizons were measured
    (``0.85``, ``0.67``, ``6.0``): with **both** energy and enstrophy and a pilot
    of ``0.2`` (one thirtieth of the run), the verdict "readable to horizon 6?"
    was correct on the two cases where the answer was no, and conservative (a
    false *no*) on the one where it was yes; horizon estimates were within
    ``1.5×`` on the hard cases.  With energy **alone** and the same pilot it
    produced a false *yes* on one pair (predicted ``8.0`` against a measured
    ``0.67``), because enstrophy is the more sensitive observable and its
    disagreement grows first.  Treat it as a *screen*: "this run will not reach
    the horizon you want" is reliable given sensitive observables, "this run
    will" is weaker, and neither replaces actually running the pair.

    Parameters:
        tolerance:        Readability tolerance (relative), default 2 %.
        integrator_order: ``q`` such that local error scales as ``dt^q``
                          (4 for classical RK4).
        min_pilot_points: Minimum envelope points required to fit at all.
    """

    def __init__(
        self,
        tolerance: float = 0.02,
        integrator_order: int = 4,
        min_pilot_points: int = 4,
    ):
        if not 0.0 < tolerance < 1.0:
            raise ValueError("tolerance must be in (0, 1)")
        if integrator_order < 1:
            raise ValueError("integrator_order must be >= 1")
        self.tolerance = tolerance
        self.integrator_order = integrator_order
        self.min_pilot_points = min_pilot_points

    @staticmethod
    def disagreement(
        pilot_dt: list[tuple[float, float]],
        pilot_half_dt: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Relative disagreement ``|a − b| / |b|`` at shared times ``t > 0``.

        Both inputs are ``(t, value)`` sequences; times are matched after
        rounding to 9 decimals so that ``0.1`` and ``0.1000000001`` coincide.
        """
        ref = {round(t, 9): v for t, v in pilot_half_dt}
        out: list[tuple[float, float]] = []
        for t, a in pilot_dt:
            if t <= 0.0:
                continue
            b = ref.get(round(t, 9))
            if b is None:
                continue
            out.append((t, abs(a - b) / max(abs(b), 1e-300)))
        return out

    @classmethod
    def combined_disagreement(
        cls,
        pairs: list[tuple[list[tuple[float, float]], list[tuple[float, float]]]],
    ) -> list[tuple[float, float]]:
        """Pointwise **maximum** relative disagreement over several observables.

        A reading rule of the form "trusted only where *both* E and Z agree
        within tolerance" is a rule on the maximum; fitting any single observable
        can miss the one that dephases first.
        """
        merged: dict[float, float] = {}
        for a, b in pairs:
            for t, e in cls.disagreement(a, b):
                k = round(t, 9)
                merged[k] = max(merged.get(k, 0.0), e)
        return sorted(merged.items())

    def predict_all(
        self,
        pairs: list[tuple[list[tuple[float, float]], list[tuple[float, float]]]],
        current_dt: float,
        target_horizon: float,
        pilot_horizon: float | None = None,
    ) -> HorizonPrediction:
        """:meth:`predict` on the combined (max) disagreement of several observables.

        Args:
            pairs:  ``[(series_at_dt, series_at_half_dt), ...]``, one per observable.
        """
        d = self.combined_disagreement(pairs)
        # Re-express as a synthetic single pair so the fitting path is shared.
        return self.predict(
            pilot_dt=[(t, 1.0 + e) for t, e in d],
            pilot_half_dt=[(t, 1.0) for t, _ in d],
            current_dt=current_dt,
            target_horizon=target_horizon,
            pilot_horizon=pilot_horizon,
        )

    def _fit_envelope(
        self, d: list[tuple[float, float]], pilot_horizon: float
    ) -> tuple[float, float, int] | None:
        """Log–log least squares on the running-max envelope over ``t <= pilot``.

        Returns ``(p, log C, n_points)`` or ``None`` if too few points.
        """
        env: list[tuple[float, float]] = []
        m = 0.0
        for t, e in d:
            if t > pilot_horizon:
                break
            m = max(m, e)
            if m > 0.0:
                env.append((t, m))
        if len(env) < self.min_pilot_points:
            return None
        xs = [math.log(t) for t, _ in env]
        ys = [math.log(e) for _, e in env]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx <= 0.0:
            return None
        p = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        return p, my - p * mx, n

    def predict(
        self,
        pilot_dt: list[tuple[float, float]],
        pilot_half_dt: list[tuple[float, float]],
        current_dt: float,
        target_horizon: float,
        pilot_horizon: float | None = None,
    ) -> HorizonPrediction:
        """Predict the readable horizon and the step needed for *target_horizon*.

        Args:
            pilot_dt:        ``(t, value)`` series from the run at step ``dt``.
            pilot_half_dt:   The same observable from the run at ``dt/2``.
            current_dt:      The step ``dt`` of the first series.
            target_horizon:  Horizon the caller wants to read up to.
            pilot_horizon:   Use only ``t <= pilot_horizon`` for the fit
                             (default: everything supplied).

        Returns:
            A :class:`HorizonPrediction`.  Never raises on poor data; low
            confidence and an explanatory ``reasoning`` are the failure mode.
        """
        d = self.disagreement(pilot_dt, pilot_half_dt)
        if pilot_horizon is None:
            pilot_horizon = d[-1][0] if d else 0.0
        fit = self._fit_envelope(d, pilot_horizon)
        if fit is None:
            return HorizonPrediction(
                readable_horizon=0.0,
                growth_exponent=0.0,
                target_readable=False,
                suggested_dt=current_dt,
                pilot_horizon=pilot_horizon,
                confidence=0.0,
                reasoning=(
                    f"Too few usable pilot points (< {self.min_pilot_points}) to fit "
                    "an error envelope; run a longer pilot."
                ),
            )
        p, log_c, n = fit
        q = self.integrator_order
        if p <= 0.0:
            # Disagreement not growing: nothing in the pilot says the target is unreadable.
            return HorizonPrediction(
                readable_horizon=math.inf,
                growth_exponent=p,
                target_readable=True,
                suggested_dt=current_dt,
                pilot_horizon=pilot_horizon,
                confidence=0.4,
                reasoning=(
                    f"Fitted growth exponent p = {p:.2f} <= 0 on {n} envelope points; "
                    "no evidence the target is unreadable, but a flat pilot is weak evidence "
                    "either way."
                ),
            )
        t_star = math.exp((math.log(self.tolerance) - log_c) / p)
        env_at_target = math.exp(log_c + p * math.log(target_horizon))
        target_ok = env_at_target <= self.tolerance
        if target_ok:
            dt_new = current_dt
        else:
            dt_new = current_dt * (self.tolerance / env_at_target) ** (1.0 / q)
        # Confidence: how much of the predicted horizon the pilot actually covered.
        coverage = min(1.0, pilot_horizon / max(t_star, 1e-300))
        confidence = 0.5 + 0.4 * coverage
        return HorizonPrediction(
            readable_horizon=t_star,
            growth_exponent=p,
            target_readable=target_ok,
            suggested_dt=dt_new,
            pilot_horizon=pilot_horizon,
            confidence=confidence,
            reasoning=(
                f"Envelope fit disagreement ≈ C·t^{p:.2f} on {n} points up to t={pilot_horizon:g}; "
                f"predicted readable horizon t ≈ {t_star:.3g} at dt={current_dt:g} "
                f"(tolerance {self.tolerance:g}). "
                + (
                    f"Target {target_horizon:g} is predicted readable."
                    if target_ok
                    else f"Target {target_horizon:g} is NOT predicted readable; with an order-{q} "
                         f"integrator, dt ≈ {dt_new:.3g} should be. Conservative estimate: it "
                         "under-predicts when error growth later saturates."
                )
            ),
        )
