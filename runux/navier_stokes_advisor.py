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
    ) -> TimeStepSuggestion:
        """Suggest an adaptive time step honouring the CFL stability limit.

        The proposed ``dt`` is the largest step satisfying:

        .. math::
            \\text{CFL} = u_{\\max} \\, \\Delta t / \\Delta x \\le \\text{cfl\\_limit}

        where ``Δx`` is estimated from the viscous Kolmogorov scale
        ``(ν³ / ε)^{1/4}`` with ε ≈ ν · u_max².

        Args:
            current_dt:    Current time step.
            cfl_number:    Current CFL number of the simulation.
            max_velocity:  Maximum velocity magnitude in the domain.
            viscosity:     Kinematic viscosity ``ν``.

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
        epsilon = viscosity * max_velocity ** 2
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
            confidence=confidence,
            reasoning=reasoning,
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
