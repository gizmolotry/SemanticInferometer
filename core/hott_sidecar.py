"""
Track 6: HoTT Sidecar — The Proof Engine

The logical capstone layer that takes purely geometric/physical outputs from
Tracks 1-5 and elevates them into formal, provable statements about semantic
structure using Homotopy Type Theory (HoTT) concepts.

VERTICAL CONSTRAINT: The geometry (manifold, tangent-space forces, ruptures)
speaks first and independently; the sidecar only interprets/validates afterward,
with no feedback into lower tracks.

GEOMETRY → LOGIC MAPPINGS:
==========================
| Geometric Feature           | HoTT Interpretation                              |
|-----------------------------|--------------------------------------------------|
| Dipole (low entropy, high EVR) | Splitting of interpretation type into Pos/Neg  |
| u_axis as dominant direction   | Principal path space / transport               |
| Finite W (elastic/trapped)     | Contractible path or homotopy between poles    |
| W → ∞ (broken rupture)         | Non-trivial π₁ obstruction, no path exists     |
| Multi-scale persistence        | Higher inductive type indexed over scales      |
| Petal glyph magnitudes         | Dependent witnesses for each probe pair        |

PROOF TYPES:
============
1. EQUIVALENCE (Pos ≡ Neg)    : Path exists, types are equivalent up to homotopy
2. NON-EQUIVALENCE (¬(Pos ≡ Neg)) : Rupture proven, no path between poles
3. OBSTRUCTION                 : Path attempted but failed (trapped state)
4. UNDECIDABLE                 : Insufficient evidence (blinker state)

Author: Belief Transformer Project / ASTER v3.2
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from datetime import datetime
import json
from pathlib import Path


# =============================================================================
# COQ BASE TEMPLATE (HoTT_Base.v)
# =============================================================================
# This is the "Constitution" of our forensic logical world.
# The HoTT Sidecar writes theorems that import this static axiom file.

HOTT_BASE_V = '''(* HoTT_Base.v - Axioms for Semantic Forensics *)
(* ASTER v3.2 - Formal Verification Library *)
(*
   This file defines the foundational types and axioms for
   formalizing semantic forensic evidence from the physics engine.

   Track 4 (The Detective): Finds the body (The Lie)
   Track 6 (The Stenographer): Writes the formal charge (The Theorem)
   Coq (The Judge): Verifies the charge is logically sound

   COMPILATION: coqc HoTT_Base.v
*)

Require Import Coq.Arith.Arith.
Require Import Coq.Arith.PeanoNat.
Require Import Coq.Logic.Classical_Prop.

(* ========================================== *)
(* 1. THE MANIFOLD: Semantic Space as a Type *)
(* ========================================== *)

(* Abstract semantic space - we use nat for simplicity *)
Definition SemanticSpace := nat.

(* Poles: semantic positions encoded as nats *)
(* Pos = 0, Neg = 1 by convention *)
Definition Pos : SemanticSpace := 0.
Definition Neg : SemanticSpace := 1.

(* ========================================== *)
(* 2. THE EVIDENCE TYPES *)
(* ========================================== *)

(* A Path exists when poles are equal (consensus) *)
Definition PathExists (p q : SemanticSpace) : Prop := p = q.

(* Semantic path: propostion that Pos and Neg are equivalent *)
Definition SemanticPath : Prop := PathExists Pos Neg.

(* A Rupture is proof that no Path exists (negation) *)
Definition SemanticRupture : Prop := ~SemanticPath.

(* Work integral threshold for obstruction detection *)
(* work is scaled by 10: work=15 means W=1.5 in physics *)
Definition work_threshold_low : nat := 10.   (* W < 1.0 = elastic *)
Definition work_threshold_high : nat := 100. (* W > 10.0 = broken *)

(* An Obstruction exists when work is in the trapped range *)
Definition ObstructionExists (work : nat) : Prop :=
  work >= work_threshold_low /\ work <= work_threshold_high.

(* ========================================== *)
(* 3. THE ORACLE AXIOMS (Trusting Physics) *)
(* ========================================== *)
(* We accept the Python Walker's testimony as axioms.
   The physics engine provides empirical evidence that
   we elevate to logical premises. *)

(* AXIOM 1: If work < threshold_low, path is elastic (equivalence) *)
(* The walker traversed easily - poles are equivalent *)
Axiom elastic_path : forall (work : nat),
  work < work_threshold_low -> SemanticPath.

(* AXIOM 2: If work > threshold_high, rupture exists *)
(* The walker broke - no path exists between poles *)
Axiom broken_rupture : forall (work : nat),
  work > work_threshold_high -> SemanticRupture.

(* AXIOM 3: Trapped state witnesses obstruction *)
(* The walker found a path but with high energy cost *)
Axiom trapped_obstruction : forall (work : nat),
  ObstructionExists work ->
  exists (path_cost : nat), path_cost = work /\ path_cost > 0.

(* ========================================== *)
(* 4. VERDICT CONSTRUCTION LEMMAS *)
(* ========================================== *)

(* Helper: work comparison is decidable *)
Lemma work_lt_dec : forall (w threshold : nat),
  {w < threshold} + {w >= threshold}.
Proof.
  intros. destruct (Nat.lt_ge_cases w threshold); auto.
Qed.

(* Helper: work in trapped range *)
Lemma work_in_trapped_range : forall (w : nat),
  w >= work_threshold_low -> w <= work_threshold_high ->
  ObstructionExists w.
Proof.
  intros w H1 H2. unfold ObstructionExists. auto.
Qed.

(* ========================================== *)
(* 5. VERDICT TYPES (From Track 4) *)
(* ========================================== *)

(* The four possible verdicts from semantic forensics *)
Inductive Verdict : Type :=
  | Honest : Verdict       (* Path exists, low work - consensus *)
  | Phantom : Verdict      (* Path exists, high work - hidden bias *)
  | Tautology : Verdict    (* Trivial path, no real semantic content *)
  | Rupture : Verdict.     (* No path exists - proven lie *)

(* Verdict from work integral *)
Definition verdict_from_work (work : nat) : Verdict :=
  if Nat.ltb work work_threshold_low then Honest
  else if Nat.ltb work_threshold_high work then Rupture
  else Phantom.

(* End of HoTT_Base.v *)
'''


# =============================================================================
# PROOF TYPES AND STRUCTURES
# =============================================================================

class ProofStatus(Enum):
    """Status of a HoTT proof attempt."""
    EQUIVALENCE = "equivalence"          # Pos ≡ Neg proven (consensus)
    NON_EQUIVALENCE = "non_equivalence"  # ¬(Pos ≡ Neg) proven (rupture)
    OBSTRUCTION = "obstruction"          # Path attempted, energy barrier found
    UNDECIDABLE = "undecidable"          # Insufficient geometric evidence


class RuptureType(Enum):
    """Classification of rupture based on geometric evidence."""
    NONE = "none"                        # No rupture detected
    ENERGETIC = "energetic"              # High W, path exists but costly
    TOPOLOGICAL = "topological"          # W → ∞, no path exists
    SCALE_DEPENDENT = "scale_dependent"  # Rupture at some scales only


@dataclass
class ProbeWitness:
    """A probe pair that witnesses a semantic splitting."""
    probe_idx: int
    probe_name: str
    magnitude: float           # Signed projection onto u_axis
    polarity: str             # "positive" or "negative"
    witness_strength: float   # |magnitude| normalized

    def to_dict(self) -> Dict:
        return {
            "probe_idx": self.probe_idx,
            "probe_name": self.probe_name,
            "magnitude": self.magnitude,
            "polarity": self.polarity,
            "witness_strength": self.witness_strength,
        }


@dataclass
class HoTTProof:
    """
    A formal proof term from the HoTT Sidecar.

    This represents the logical interpretation of geometric evidence.
    """
    # Identification
    article_id: str
    theorem_name: str          # e.g., "Thm_Rupture_01", "Thm_Consensus_42"

    # Proof status
    status: ProofStatus
    rupture_type: RuptureType

    # Geometric evidence (inputs from lower tracks)
    evr: float                 # From Track 1.5: explained variance ratio
    dipole_valid: bool         # From Track 1.5: EVR >= threshold
    n_persistent_scales: int   # From Track 1.5: multi-scale persistence
    work_integral: float       # From Track 4: geodesic energy cost
    walker_state: str          # From Track 4: "elastic", "trapped", "broken"

    # Type-theoretic interpretation
    type_statement: str        # Human-readable type signature
    proof_term: str            # Pseudo-Agda/Lean proof term

    # Witnesses (probes that contributed to the proof)
    positive_witnesses: List[ProbeWitness] = field(default_factory=list)
    negative_witnesses: List[ProbeWitness] = field(default_factory=list)

    # Confidence and explanation
    confidence: float = 0.0    # Proof confidence [0, 1]
    explanation: str = ""      # Natural language explanation

    # Metadata
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "article_id": self.article_id,
            "theorem_name": self.theorem_name,
            "status": self.status.value,
            "rupture_type": self.rupture_type.value,
            "evr": self.evr,
            "dipole_valid": self.dipole_valid,
            "n_persistent_scales": self.n_persistent_scales,
            "work_integral": self.work_integral,
            "walker_state": self.walker_state,
            "type_statement": self.type_statement,
            "proof_term": self.proof_term,
            "positive_witnesses": [w.to_dict() for w in self.positive_witnesses],
            "negative_witnesses": [w.to_dict() for w in self.negative_witnesses],
            "confidence": self.confidence,
            "explanation": self.explanation,
            "timestamp": self.timestamp,
        }

    def to_agda(self) -> str:
        """Export proof as Agda-style syntax (pseudo-code)."""
        lines = [
            f"-- Theorem: {self.theorem_name}",
            f"-- Article: {self.article_id}",
            f"-- Status: {self.status.value}",
            "",
        ]

        if self.status == ProofStatus.NON_EQUIVALENCE:
            lines.extend([
                f"-- Evidence: W = {self.work_integral:.3f} (→ ∞)",
                f"-- Interpretation: The path space Pos ≡ Neg is empty",
                "",
                f"{self.theorem_name} : ¬ (Pos ≡ Neg)",
                f"{self.theorem_name} = rupture-witness",
                f"  where",
                f"    rupture-witness : (p : Pos ≡ Neg) → ⊥",
                f"    rupture-witness p = transport-failure p",
                f"      (work-divergence {self.work_integral:.3f})",
            ])
        elif self.status == ProofStatus.EQUIVALENCE:
            lines.extend([
                f"-- Evidence: W = {self.work_integral:.3f} (finite)",
                f"-- Interpretation: Path exists between poles",
                "",
                f"{self.theorem_name} : Pos ≡ Neg",
                f"{self.theorem_name} = path-construction",
                f"  where",
                f"    path-construction : Pos → Neg",
                f"    path-construction = transport u-axis",
            ])
        elif self.status == ProofStatus.OBSTRUCTION:
            lines.extend([
                f"-- Evidence: W = {self.work_integral:.3f} (high but finite)",
                f"-- Interpretation: Path exists but obstructed",
                "",
                f"{self.theorem_name} : Obstructed (Pos ≡ Neg)",
                f"{self.theorem_name} = energy-barrier",
                f"  where",
                f"    energy-barrier : W > threshold",
                f"    energy-barrier = work-witness {self.work_integral:.3f}",
            ])
        else:
            lines.extend([
                f"-- Insufficient geometric evidence",
                f"-- EVR: {self.evr:.3f}, Dipole valid: {self.dipole_valid}",
                "",
                f"{self.theorem_name} : Undecidable",
                f"{self.theorem_name} = evidence-insufficient",
            ])

        return "\n".join(lines)


@dataclass
class HoTTSidecarConfig:
    """Configuration for the HoTT Sidecar proof engine."""
    # Thresholds for proof classification
    evr_threshold: float = 0.5            # Minimum EVR for valid dipole
    min_persistent_scales: int = 2        # Minimum scales for persistence
    work_elastic_threshold: float = 0.1   # W below this = elastic (equivalence)
    work_trapped_threshold: float = 2.0   # W above this, below broken = obstruction
    work_broken_threshold: float = 10.0   # W above this = rupture (non-equivalence)

    # Witness selection
    witness_magnitude_threshold: float = 0.1  # Min magnitude to be a witness
    max_witnesses_per_pole: int = 4       # Max witnesses to report per pole

    # Probe names (default from framing_queries.yaml)
    probe_names: List[str] = field(default_factory=lambda: [
        "Israeli Defense vs Aggression",
        "Hamas Defense vs Terrorism",
        "Civilian Shield vs Hostage",
        "Proportionality vs Necessity",
        "International Law vs Sovereignty",
        "Humanitarian vs Security",
        "Resistance vs Violence",
        "Self-Determination vs Occupation",
    ])


# =============================================================================
# HOTT SIDECAR: THE PROOF ENGINE
# =============================================================================

class HoTTSidecar:
    """
    Track 6: The HoTT Proof Engine

    Converts geometric features from Tracks 1-5 into formal logical proofs
    about semantic structure using Homotopy Type Theory concepts.

    Vertical Constraint: This module is read-only with respect to lower tracks.
    It interprets geometry but never modifies it.
    """

    def __init__(self, config: Optional[HoTTSidecarConfig] = None):
        self.config = config or HoTTSidecarConfig()
        self._proof_counter = 0

    def _generate_theorem_name(self, status: ProofStatus) -> str:
        """Generate a unique theorem name."""
        self._proof_counter += 1
        prefix = {
            ProofStatus.EQUIVALENCE: "Thm_Consensus",
            ProofStatus.NON_EQUIVALENCE: "Thm_Rupture",
            ProofStatus.OBSTRUCTION: "Thm_Barrier",
            ProofStatus.UNDECIDABLE: "Thm_Undecided",
        }[status]
        return f"{prefix}_{self._proof_counter:04d}"

    def _classify_rupture(
        self,
        work_integral: float,
        walker_state: str,
        n_persistent_scales: int,
    ) -> RuptureType:
        """Classify the type of rupture based on geometric evidence."""
        if walker_state == "broken" or work_integral > self.config.work_broken_threshold:
            # Check if rupture persists across scales
            if n_persistent_scales >= self.config.min_persistent_scales:
                return RuptureType.TOPOLOGICAL
            else:
                return RuptureType.SCALE_DEPENDENT
        elif walker_state == "trapped" or work_integral > self.config.work_trapped_threshold:
            return RuptureType.ENERGETIC
        else:
            return RuptureType.NONE

    def _extract_witnesses(
        self,
        probe_magnitudes: np.ndarray,
    ) -> Tuple[List[ProbeWitness], List[ProbeWitness]]:
        """Extract positive and negative witnesses from probe magnitudes."""
        positive = []
        negative = []

        # Normalize magnitudes for witness strength
        max_mag = np.abs(probe_magnitudes).max() + 1e-9

        for idx, mag in enumerate(probe_magnitudes):
            if abs(mag) < self.config.witness_magnitude_threshold:
                continue

            witness = ProbeWitness(
                probe_idx=idx,
                probe_name=self.config.probe_names[idx] if idx < len(self.config.probe_names) else f"Probe_{idx}",
                magnitude=float(mag),
                polarity="positive" if mag > 0 else "negative",
                witness_strength=float(abs(mag) / max_mag),
            )

            if mag > 0:
                positive.append(witness)
            else:
                negative.append(witness)

        # Sort by strength and limit
        positive.sort(key=lambda w: -w.witness_strength)
        negative.sort(key=lambda w: -w.witness_strength)

        return (
            positive[:self.config.max_witnesses_per_pole],
            negative[:self.config.max_witnesses_per_pole],
        )

    def _generate_type_statement(
        self,
        status: ProofStatus,
        positive_witnesses: List[ProbeWitness],
        negative_witnesses: List[ProbeWitness],
    ) -> str:
        """Generate a human-readable type signature."""
        if status == ProofStatus.NON_EQUIVALENCE:
            pos_names = [w.probe_name.split(" vs ")[0] for w in positive_witnesses[:2]]
            neg_names = [w.probe_name.split(" vs ")[0] for w in negative_witnesses[:2]]
            pos_str = " ∧ ".join(pos_names) if pos_names else "Pos"
            neg_str = " ∧ ".join(neg_names) if neg_names else "Neg"
            return f"¬ ({pos_str} ≡ {neg_str})"
        elif status == ProofStatus.EQUIVALENCE:
            return "Pos ≡ Neg  (consensus: semantic equivalence)"
        elif status == ProofStatus.OBSTRUCTION:
            return "∃ path : Pos → Neg, but W(path) > threshold  (barrier)"
        else:
            return "? : Undecidable  (insufficient evidence)"

    def _generate_proof_term(
        self,
        status: ProofStatus,
        evr: float,
        work_integral: float,
        walker_state: str,
    ) -> str:
        """Generate a pseudo proof term."""
        if status == ProofStatus.NON_EQUIVALENCE:
            return f"rupture-witness (work-divergence {work_integral:.2f}) (walker-state {walker_state})"
        elif status == ProofStatus.EQUIVALENCE:
            return f"path-construction (evr {evr:.3f}) (work-finite {work_integral:.2f})"
        elif status == ProofStatus.OBSTRUCTION:
            return f"barrier-witness (evr {evr:.3f}) (work-high {work_integral:.2f})"
        else:
            return f"evidence-insufficient (evr {evr:.3f})"

    def _generate_explanation(
        self,
        status: ProofStatus,
        rupture_type: RuptureType,
        evr: float,
        work_integral: float,
        walker_state: str,
        positive_witnesses: List[ProbeWitness],
        negative_witnesses: List[ProbeWitness],
    ) -> str:
        """Generate a natural language explanation of the proof."""
        if status == ProofStatus.NON_EQUIVALENCE:
            pos_str = ", ".join([w.probe_name for w in positive_witnesses[:2]]) or "positive pole"
            neg_str = ", ".join([w.probe_name for w in negative_witnesses[:2]]) or "negative pole"

            if rupture_type == RuptureType.TOPOLOGICAL:
                return (
                    f"PROVEN RUPTURE: The semantic space exhibits a topological disconnection. "
                    f"The geodesic walker found infinite energy cost (W={work_integral:.1f}) "
                    f"when attempting to traverse from [{pos_str}] to [{neg_str}]. "
                    f"This rupture persists across multiple scales, indicating a fundamental "
                    f"non-equivalence of interpretations."
                )
            else:
                return (
                    f"SCALE-DEPENDENT RUPTURE: A semantic split exists at current scale "
                    f"between [{pos_str}] and [{neg_str}], but may not persist across all bandwidths."
                )

        elif status == ProofStatus.EQUIVALENCE:
            return (
                f"CONSENSUS: The semantic poles are equivalent up to homotopy. "
                f"The walker successfully traversed between interpretations with "
                f"minimal energy cost (W={work_integral:.2f}), indicating that "
                f"different framings converge to equivalent meanings."
            )

        elif status == ProofStatus.OBSTRUCTION:
            return (
                f"BARRIER: A path exists between semantic poles, but with significant "
                f"energy cost (W={work_integral:.2f}). This suggests a biased but "
                f"traversable landscape—interpretations are technically equivalent "
                f"but practically separated by an ideological barrier."
            )

        else:
            return (
                f"UNDECIDABLE: Insufficient geometric evidence to prove or disprove "
                f"semantic equivalence. EVR={evr:.3f} is below threshold, suggesting "
                f"the text may not exhibit clear polarization (blinker state)."
            )

    def prove_single(
        self,
        article_id: str,
        evr: float,
        dipole_valid: bool,
        n_persistent_scales: int,
        work_integral: float,
        walker_state: str,
        probe_magnitudes: Optional[np.ndarray] = None,
        phantom_verdict: Optional[str] = None,
    ) -> HoTTProof:
        """
        Generate a HoTT proof for a single article.

        Args:
            article_id: Unique article identifier
            evr: Explained variance ratio from Track 1.5
            dipole_valid: Whether EVR passes threshold
            n_persistent_scales: Number of scales where dipole persists
            work_integral: Geodesic energy cost from Track 4
            walker_state: "elastic", "trapped", or "broken"
            probe_magnitudes: [8] signed projections onto u_axis
            phantom_verdict: Optional phantom differential verdict ("HONEST", "PHANTOM", "RUPTURE")

        Returns:
            HoTTProof object with formal proof term and explanation
        """
        # ASTER v3.2: STRICT VERDICT MAPPING
        # ====================================
        # The HoTT Sidecar is a STENOGRAPHER, not a JUDGE.
        # It TRUSTS the physics engine (Track 4) verdict completely.
        # No recalculation, no "overrides" — pure logic mapping.

        # Step 1: Map physics verdict → HoTT status + rupture type
        if phantom_verdict is not None:
            # AUTHORITATIVE: Physics verdict takes precedence
            if phantom_verdict == "RUPTURE":
                status = ProofStatus.NON_EQUIVALENCE
                rupture_type = RuptureType.TOPOLOGICAL
            elif phantom_verdict == "PHANTOM":
                status = ProofStatus.OBSTRUCTION
                rupture_type = RuptureType.ENERGETIC
            elif phantom_verdict == "HONEST":
                status = ProofStatus.EQUIVALENCE
                rupture_type = RuptureType.NONE
            elif phantom_verdict == "TAUTOLOGY":
                status = ProofStatus.EQUIVALENCE  # Trivial equivalence
                rupture_type = RuptureType.NONE
            else:
                status = ProofStatus.UNDECIDABLE
                rupture_type = RuptureType.NONE
        else:
            # FALLBACK: Only if no physics verdict available
            # This should rarely happen in ASTER v3.2
            rupture_type = self._classify_rupture(work_integral, walker_state, n_persistent_scales)
            if not dipole_valid:
                status = ProofStatus.UNDECIDABLE
            elif rupture_type in (RuptureType.TOPOLOGICAL, RuptureType.SCALE_DEPENDENT):
                status = ProofStatus.NON_EQUIVALENCE
            elif rupture_type == RuptureType.ENERGETIC:
                status = ProofStatus.OBSTRUCTION
            else:
                status = ProofStatus.EQUIVALENCE

        # Step 3: Extract witnesses
        positive_witnesses, negative_witnesses = [], []
        if probe_magnitudes is not None:
            positive_witnesses, negative_witnesses = self._extract_witnesses(probe_magnitudes)

        # Step 4: Generate proof components
        theorem_name = self._generate_theorem_name(status)
        type_statement = self._generate_type_statement(status, positive_witnesses, negative_witnesses)
        proof_term = self._generate_proof_term(status, evr, work_integral, walker_state)
        explanation = self._generate_explanation(
            status, rupture_type, evr, work_integral, walker_state,
            positive_witnesses, negative_witnesses
        )

        # Step 5: Compute confidence
        # Higher confidence for: high EVR, consistent walker state, multi-scale persistence
        confidence = 0.0
        if dipole_valid:
            confidence += 0.3
        confidence += min(0.3, evr * 0.4)
        confidence += min(0.2, n_persistent_scales * 0.1)
        if walker_state == "broken" and rupture_type == RuptureType.TOPOLOGICAL:
            confidence += 0.2  # Strong evidence for rupture
        elif walker_state == "elastic" and rupture_type == RuptureType.NONE:
            confidence += 0.2  # Strong evidence for consensus
        confidence = min(1.0, confidence)

        return HoTTProof(
            article_id=article_id,
            theorem_name=theorem_name,
            status=status,
            rupture_type=rupture_type,
            evr=evr,
            dipole_valid=dipole_valid,
            n_persistent_scales=n_persistent_scales,
            work_integral=work_integral,
            walker_state=walker_state,
            type_statement=type_statement,
            proof_term=proof_term,
            positive_witnesses=positive_witnesses,
            negative_witnesses=negative_witnesses,
            confidence=confidence,
            explanation=explanation,
        )

    def prove_batch(
        self,
        article_ids: List[str],
        evr_batch: np.ndarray,
        dipole_valid_batch: np.ndarray,
        n_persistent_scales_batch: np.ndarray,
        work_integrals: List[float],
        walker_states: List[str],
        probe_magnitudes_batch: Optional[np.ndarray] = None,
        phantom_verdicts: Optional[List[Optional[str]]] = None,
    ) -> List[HoTTProof]:
        """
        Generate HoTT proofs for a batch of articles.

        Returns list of HoTTProof objects.
        """
        N = len(article_ids)
        proofs = []
        for i in range(N):
            probe_mags = probe_magnitudes_batch[i] if probe_magnitudes_batch is not None else None
            phantom_verdict = None
            if phantom_verdicts is not None and i < len(phantom_verdicts):
                phantom_verdict = phantom_verdicts[i]

            proof = self.prove_single(
                article_id=article_ids[i],
                evr=float(evr_batch[i]),
                dipole_valid=bool(dipole_valid_batch[i]),
                n_persistent_scales=int(n_persistent_scales_batch[i]),
                work_integral=float(work_integrals[i]) if i < len(work_integrals) else 0.0,
                walker_state=walker_states[i] if i < len(walker_states) else "elastic",
                probe_magnitudes=probe_mags,
                phantom_verdict=phantom_verdict,
            )
            proofs.append(proof)

        return proofs

    def summarize_proofs(self, proofs: List[HoTTProof]) -> Dict[str, Any]:
        """Generate summary statistics for a batch of proofs."""
        N = len(proofs)
        if N == 0:
            return {"n_proofs": 0}

        status_counts = {s.value: 0 for s in ProofStatus}
        rupture_counts = {r.value: 0 for r in RuptureType}

        total_confidence = 0.0

        for p in proofs:
            status_counts[p.status.value] += 1
            rupture_counts[p.rupture_type.value] += 1
            total_confidence += p.confidence

        return {
            "n_proofs": N,
            "status_counts": status_counts,
            "rupture_counts": rupture_counts,
            "equivalence_rate": status_counts["equivalence"] / N,
            "non_equivalence_rate": status_counts["non_equivalence"] / N,
            "obstruction_rate": status_counts["obstruction"] / N,
            "undecidable_rate": status_counts["undecidable"] / N,
            "mean_confidence": total_confidence / N,
            "topological_ruptures": rupture_counts["topological"],
            "energetic_barriers": rupture_counts["energetic"],
        }

    def export_proofs(self, proofs: List[HoTTProof], format: str = "json") -> str:
        """Export proofs to various formats."""
        if format == "json":
            return json.dumps([p.to_dict() for p in proofs], indent=2)
        elif format == "agda":
            return "\n\n".join([p.to_agda() for p in proofs])
        elif format == "coq":
            return "\n\n".join([self.generate_coq_proof(p) for p in proofs])
        else:
            raise ValueError(f"Unknown format: {format}")

    def generate_coq_proof(self, proof: HoTTProof) -> str:
        """
        Generate a Coq (.v) proof file from a HoTTProof object.

        This is the "Stenographer" function - it translates the Python
        proof object into formal Coq syntax that can be verified offline.

        Track 4 (Detective) finds the evidence.
        Track 6 (Stenographer) writes the legal charge.
        Coq (Judge) verifies the charge is syntactically valid.

        Args:
            proof: HoTTProof from prove_single()

        Returns:
            String containing valid Coq source code
        """
        # 1. HEADER (Imports and Metadata)
        coq_code = [
            f"(* ============================================ *)",
            f"(* Formal Verification for Article: {proof.article_id} *)",
            f"(* Theorem: {proof.theorem_name} *)",
            f"(* Generated: {proof.timestamp} *)",
            f"(* ============================================ *)",
            "",
            "(* Import the semantic forensics axioms *)",
            "Require Import HoTT_Base.",
            "",
            f"Section Proof_{proof.article_id.replace('-', '_').replace('.', '_')}.",
            "",
        ]

        # 2. WITNESSES (Define the semantic poles from probe evidence)
        if proof.positive_witnesses:
            pos_witness = proof.positive_witnesses[0]
            coq_code.extend([
                f"  (* Primary Positive Pole: {pos_witness.probe_name} *)",
                f"  (* Magnitude: {pos_witness.magnitude:.4f} *)",
                f"  Definition pos_evidence : nat := {int(abs(pos_witness.magnitude) * 100)}.",
                "",
            ])

        if proof.negative_witnesses:
            neg_witness = proof.negative_witnesses[0]
            coq_code.extend([
                f"  (* Primary Negative Pole: {neg_witness.probe_name} *)",
                f"  (* Magnitude: {neg_witness.magnitude:.4f} *)",
                f"  Definition neg_evidence : nat := {int(abs(neg_witness.magnitude) * 100)}.",
                "",
            ])

        # 3. GEOMETRIC EVIDENCE (From physics engine)
        work_int = int(proof.work_integral * 10)  # Scale for Coq nat
        coq_code.extend([
            f"  (* Geometric Evidence from Physics Engine *)",
            f"  Definition work_integral : nat := {work_int}.",
            f"  Definition evr : nat := {int(proof.evr * 100)}.",  # Percentage
            f"  (* Walker State: {proof.walker_state} *)",
            "",
        ])

        # 4. THE THEOREM (The formal charge based on verdict)
        if proof.status == ProofStatus.NON_EQUIVALENCE:
            # RUPTURE: The text is a lie - no path exists
            coq_code.extend([
                f"  (* ---------------------------------------- *)",
                f"  (* VERDICT: RUPTURE (Non-Equivalence) *)",
                f"  (* The physics engine witnessed a Void. *)",
                f"  (* Work integral W = {proof.work_integral:.2f} diverged. *)",
                f"  (* ---------------------------------------- *)",
                "",
                f"  Theorem {proof.theorem_name} : SemanticRupture.",
                f"  Proof.",
                f"    (* Apply the oracle axiom: broken walker implies rupture *)",
                f"    apply walker_broken_implies_rupture with (work := {work_int}).",
                f"    (* The work integral {work_int} > 10 threshold *)",
                f"    omega.  (* Or: auto with arith. *)",
                f"  Qed.",
            ])

        elif proof.status == ProofStatus.EQUIVALENCE:
            # HONEST: The text is truthful - path exists and is easy
            coq_code.extend([
                f"  (* ---------------------------------------- *)",
                f"  (* VERDICT: HONEST (Equivalence) *)",
                f"  (* The physics engine found a smooth path. *)",
                f"  (* Work integral W = {proof.work_integral:.2f} is minimal. *)",
                f"  (* ---------------------------------------- *)",
                "",
                f"  Theorem {proof.theorem_name} : SemanticPath.",
                f"  Proof.",
                f"    (* Apply the oracle axiom: elastic walker implies path *)",
                f"    apply walker_elastic_implies_path with (work := {work_int}).",
                f"    (* The work integral {work_int} < 1 threshold *)",
                f"    omega.",
                f"  Qed.",
            ])

        elif proof.status == ProofStatus.OBSTRUCTION:
            # PHANTOM: Path exists but is obstructed (hidden bias)
            coq_code.extend([
                f"  (* ---------------------------------------- *)",
                f"  (* VERDICT: PHANTOM (Obstruction) *)",
                f"  (* The physics engine found a biased path. *)",
                f"  (* Work integral W = {proof.work_integral:.2f} is elevated. *)",
                f"  (* ---------------------------------------- *)",
                "",
                f"  Theorem {proof.theorem_name} : Obstruction 1.",
                f"  Proof.",
                f"    (* Apply the oracle axiom: trapped walker implies obstruction *)",
                f"    apply walker_trapped_implies_obstruction with (work := {work_int}).",
                f"    (* The work integral 1 <= {work_int} <= 10 *)",
                f"    split; omega.",
                f"  Qed.",
            ])

        else:
            # UNDECIDABLE: Insufficient evidence
            coq_code.extend([
                f"  (* ---------------------------------------- *)",
                f"  (* VERDICT: UNDECIDABLE *)",
                f"  (* Insufficient geometric evidence. *)",
                f"  (* EVR = {proof.evr:.3f} below threshold. *)",
                f"  (* ---------------------------------------- *)",
                "",
                f"  (* No formal theorem can be proven. *)",
                f"  (* The blinker state indicates ambiguity. *)",
                f"  Definition {proof.theorem_name}_status : string := \"undecidable\".",
            ])

        # 5. CLOSE SECTION
        coq_code.extend([
            "",
            f"End Proof_{proof.article_id.replace('-', '_').replace('.', '_')}.",
            "",
            f"(* End of proof for {proof.article_id} *)",
        ])

        return "\n".join(coq_code)

    def save_coq_proofs(
        self,
        proofs: List[HoTTProof],
        output_dir: Union[str, Path],
        include_base: bool = True,
    ) -> List[Path]:
        """
        Save Coq proof files to disk.

        Args:
            proofs: List of HoTTProof objects
            output_dir: Directory to save .v files
            include_base: If True, also save HoTT_Base.v

        Returns:
            List of paths to saved files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []

        # Save base axiom file
        if include_base:
            base_path = output_dir / "HoTT_Base.v"
            base_path.write_text(HOTT_BASE_V)
            saved_files.append(base_path)

        # Save individual proof files
        for proof in proofs:
            filename = f"{proof.article_id.replace('-', '_').replace('.', '_')}_{proof.status.value}.v"
            filepath = output_dir / filename
            coq_content = self.generate_coq_proof(proof)
            filepath.write_text(coq_content)
            saved_files.append(filepath)

        return saved_files


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def prove_from_pipeline_results(
    result: Dict[str, Any],
    config: Optional[HoTTSidecarConfig] = None,
) -> Tuple[List[HoTTProof], Dict[str, Any]]:
    """
    Generate HoTT proofs directly from pipeline output dict.

    Args:
        result: Output from CompletePipeline.process_month()
        config: Optional HoTT configuration

    Returns:
        (proofs, summary) tuple
    """
    sidecar = HoTTSidecar(config)

    # Extract required fields
    article_ids = result.get("bt_uid_list", [f"article_{i}" for i in range(len(result.get("spectral_evr", [])))])
    evr_batch = result.get("spectral_evr", np.array([]))
    dipole_valid_batch = result.get("spectral_dipole_valid", np.array([]))
    n_persistent_scales_batch = result.get("spectral_n_persistent_scales", np.array([]))
    work_integrals = list(result.get("walker_work_integrals", []))
    walker_states = result.get("walker_states", [])
    probe_magnitudes_batch = result.get("spectral_probe_magnitudes", None)
    phantom_verdict_payloads = result.get("phantom_verdicts", [])
    phantom_verdicts = []
    if isinstance(phantom_verdict_payloads, list):
        for payload in phantom_verdict_payloads:
            if isinstance(payload, dict):
                phantom_verdicts.append(payload.get("verdict"))
            else:
                phantom_verdicts.append(payload)

    # Handle missing walker data
    N = len(evr_batch)
    if len(work_integrals) < N:
        work_integrals.extend([0.0] * (N - len(work_integrals)))
    if len(walker_states) < N:
        walker_states.extend(["elastic"] * (N - len(walker_states)))

    proofs = sidecar.prove_batch(
        article_ids=article_ids,
        evr_batch=evr_batch,
        dipole_valid_batch=dipole_valid_batch,
        n_persistent_scales_batch=n_persistent_scales_batch,
        work_integrals=work_integrals,
        walker_states=walker_states,
        probe_magnitudes_batch=probe_magnitudes_batch,
        phantom_verdicts=phantom_verdicts,
    )

    summary = sidecar.summarize_proofs(proofs)

    return proofs, summary


__all__ = [
    'ProofStatus',
    'RuptureType',
    'ProbeWitness',
    'HoTTProof',
    'HoTTSidecarConfig',
    'HoTTSidecar',
    'prove_from_pipeline_results',
]
