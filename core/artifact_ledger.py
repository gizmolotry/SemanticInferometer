import sys
from pathlib import Path

class ArtifactContract:
    """
    ASTER v3.2 Artifact Contract Enforcement
    Strictly verifies that all required physics payloads exist before allowing 
    downstream execution. NO FALLBACKS ALLOWED.
    """
    def __init__(self, directory: Path, require_spectral_dna: bool = False):
        self.directory = Path(directory)
        self.required_files = [
            "features.npy",
            "article_metadata.csv",
            "validation.json",
        ]
        self.require_spectral_dna = bool(require_spectral_dna)
        self.spectral_files = [
            "spectral_probe_magnitudes.npy",
        ]

    def verify(self):
        missing = [f for f in self.required_files if not (self.directory / f).exists()]
        if self.require_spectral_dna:
            missing.extend([f for f in self.spectral_files if not (self.directory / f).exists()])
        
        if missing:
            error_msg = (
                f"[CRITICAL CONTRACT VIOLATION] Target Directory: {self.directory}\n"
                f"MISSING REQUIRED ARTIFACTS: {', '.join(missing)}\n"
                f"Status: FAIL-FAST. Execution halted to prevent pipeline contamination."
            )
            print(error_msg, file=sys.stderr)
            # Raise for programmatic catch and hard-halt behavior.
            raise RuntimeError(error_msg)
        
        expected = len(self.required_files) + (len(self.spectral_files) if self.require_spectral_dna else 0)
        print(f"[CONTRACT OK] All {expected} artifacts verified in {self.directory.name}")
