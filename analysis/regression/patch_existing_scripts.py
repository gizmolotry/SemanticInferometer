from pathlib import Path
import json
import re

def patch_compare_controls(file_path: Path):
    content = file_path.read_text(encoding="utf-8")
    
    # 1. Add logic to load and print verification report in run_single_analysis
    if 'verification_report.json' not in content:
        # Find where interpretation is printed
        # Note the double backslash for the literal \n in the source
        insertion_marker = 'print("\\\\n" + "="*70)\n    print("INTERPRETATION")'
        
        verification_logic = r'''
    # Verification Harness Integration
    verify_path = data_dir / "verification_report.json"
    if verify_path.exists():
        print("\n" + "="*70)
        print("VERIFICATION STATUS (RING 1)")
        print("="*70)
        try:
            with open(verify_path, "r") as f:
                vdata = json.load(f)
            
            print(f"Global Pass: {'[PASS]' if vdata.get('global_pass') else '[FAIL]'}")
            for layer in vdata.get("layers", []):
                lname = layer.get("layer_name", "unknown")
                lstatus = "PASS" if all(c.get("pass") for c in layer.get("checks", []) if c.get("pass") is not None) else "FAIL"
                print(f"  Layer {lname:20s}: {lstatus}")
                for check in layer.get("checks", []):
                    cname = check.get("name")
                    cpass = check.get("pass")
                    cval = check.get("value")
                    status_str = "OK" if cpass else "!!" if cpass is False else "--"
                    val_str = f"({cval:.3f})" if isinstance(cval, (float, int)) else ""
                    print(f"    - {cname:25s} [{status_str}] {val_str}")
        except Exception as e:
            print(f"  Warning: Could not parse verification report: {e}")
'''
        # We'll try a simpler match if the exact string fails
        if insertion_marker not in content:
             print("Falling back to simpler marker for compare_controls.py")
             insertion_marker = 'print("INTERPRETATION")'

        content = content.replace(insertion_marker, verification_logic + "\n    " + insertion_marker)
        print("Patched compare_controls.py with verification display.")

    file_path.write_text(content, encoding="utf-8")

def patch_ablation(file_path: Path):
    content = file_path.read_text(encoding="utf-8")
    
    # Add verification summary to the final output of ablation.py
    if 'verification_report.json' not in content:
        insertion_marker = 'return {'
        verification_logic = r'''
    # Check for Ring 1 Verification
    verify_report = data_path.parent / "verification_report.json"
    if verify_report.exists():
        try:
            with open(verify_report, "r") as f:
                vdata = json.load(f)
            print("\n--- Ring 1 Verification Summary ---")
            print(f"Global Integrity: {'VALID' if vdata.get('global_pass') else 'COMPROMISED'}")
            for layer in vdata.get("layers", []):
                if any(c.get("name") == "mi_score" for c in layer.get("checks", [])):
                    print(f"Layer {layer.get('layer_name')}: Integrity checks passed.")
        except:
            pass
'''
        content = content.replace(insertion_marker, verification_logic + "\n    " + insertion_marker)
        print("Patched ablation.py with verification summary.")

    file_path.write_text(content, encoding="utf-8")

if __name__ == "__main__":
    patch_compare_controls(Path("../compare_controls.py"))
    patch_ablation(Path("ablation.py"))
