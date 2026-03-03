import sys
from pathlib import Path

class ArtifactContract:
    """
    ASTER v3.2 Artifact Contract Enforcement
    Strictly verifies that all required physics payloads exist before allowing 
    downstream execution. NO FALLBACKS ALLOWED.
    """
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.required_files = [
            "features.npy",
            "walker_states.json",
            "walker_work_integrals.npy",
            "phantom_verdicts.json",
            "article_metadata.csv"
        ]

    def verify(self):
        missing = [f for f in self.required_files if not (self.directory / f).exists()]
        
        if missing:
            error_msg = (
                f"
[CRITICAL CONTRACT VIOLATION] Target Directory: {self.directory}
"
                f"MISSING REQUIRED ARTIFACTS: {', '.join(missing)}
"
                f"Status: FAIL-FAST. Execution halted to prevent pipeline contamination."
            )
            print(error_msg, file=sys.stderr)
            # Raise for programmatic catch, though sys.exit(1) is the requested hard-halt.
            raise RuntimeError(error_msg)
            sys.exit(1)
        
        print(f"[CONTRACT OK] All {len(self.required_files)} artifacts verified in {self.directory.name}")
