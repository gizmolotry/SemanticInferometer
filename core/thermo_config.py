from dataclasses import dataclass


@dataclass
class ThermodynamicConfig:
    # Walker Dynamics (Cranked up to be deadly)
    friction_coefficient: float = 0.4  # High drag
    noise_sigma: float = 0.1
    tautology_work_threshold: float = 5.0  # Example high threshold
    tautology_disp_threshold: float = 0.2

    # Fusion & Terrain
    hadamard_floor: float = 0.1  # Softens the AND gate to rescue NMI
    density_clamp_min: float = 1e-4  # Prevents -Infinity black holes
    stress_threshold: float = 0.75  # Tightened to expose more cracks
