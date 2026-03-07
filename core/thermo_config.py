from dataclasses import dataclass


@dataclass
class ThermodynamicConfig:
    # Walker Dynamics (Cranked up to be deadly)
    friction_coefficient: float = 0.5  # High drag (sharpens cracks)
    noise_sigma: float = 0.1
    tautology_work_threshold: float = 5.0  # Example high threshold
    tautology_disp_threshold: float = 0.2

    # Fusion & Terrain
    hadamard_floor: float = 0.15  # Softens the AND gate to rescue NMI + ridge connectivity
    density_clamp_min: float = 1e-4  # Prevents -Infinity black holes
    epsilon_z: float = 1e-4  # Soft floor for z-height transform: z = -log(density + epsilon_z)
    stress_threshold: float = 0.75  # Tightened to expose more cracks

    # Walker Energy Budget (Track 4 Thermodynamics)
    walker_energy_budget: float = 50.0  # Total work units before walker dies on the mountain
