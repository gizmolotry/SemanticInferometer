import re
from pathlib import Path

def fix_monolith_viz_variables_robust():
    path = Path('MONOLITH_VIZ.py')
    content = path.read_text(encoding='utf-8')

    # Fix Family C variable: evr -> spectral_evr[i]
    content = content.replace('if evr < 0.5:', 'if spectral_evr is not None and i < len(spectral_evr) and spectral_evr[i] < 0.5:')

    # Rename 'colors' parameter to 'spectral_evr' in the function signature
    content = re.sub(r'def render_data_points_3d\(\s*positions:\s*np\.ndarray,\s*colors:\s*np\.ndarray,', 
                     'def render_data_points_3d(\n    positions: np.ndarray,\n    spectral_evr: np.ndarray,', 
                     content, flags=re.DOTALL)

    path.write_text(content, encoding='utf-8')
    print("Variable fixes applied robustly.")

if __name__ == "__main__":
    fix_monolith_viz_variables_robust()
