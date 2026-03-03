from pathlib import Path
import json
import re

# This script cleans up and standardizes the integration of verification results into compare_controls.py.
# It ensures ONE clear print block and proper status reporting.

target = Path("../compare_controls.py")
content = target.read_text(encoding="utf-8")

# 1. Locate the existing verification block(s) if any and remove them to start fresh.
# The previous bot added a block starting with "# Verification Harness Integration"
pattern = r'# Verification Harness Integration.*?except Exception as e:.*?print\(f"  Warning: Could not parse verification report: \{e\}"\)'
content = re.sub(pattern, '', content, flags=re.DOTALL)

# 2. Re-insert the standardized thesis-grade verification display.
# We'll place it right before "INTERPRETATION" in run_single_analysis.
insertion_marker = 'print("INTERPRETATION")'

verification_logic = r'''
    # =========================================================================
    # THESIS-GRADE VERIFICATION STATUS (RING 1)
    # =========================================================================
    verify_path = data_dir / "verification_report.json"
    if verify_path.exists():
        print("\n" + "="*70)
        print("VERIFICATION STATUS (RING 1)")
        print("="*70)
        try:
            with open(verify_path, "r") as f:
                vdata = json.load(f)
            
            print(f"Global Integrity Pass: {'[PASS]' if vdata.get('global_pass') else '[FAIL]'}")
            print(f"Run ID: {vdata.get('run_id', 'unknown')}")
            
            for layer in vdata.get("layers", []):
                lname = layer.get("layer_name", "unknown")
                lid = layer.get("layer_id", lname)
                lstatus = layer.get("status", "UNVERIFIED")
                
                # Visual formatting (simple text markers for terminal)
                status_icon = "[V]" if lstatus == "VERIFIED" else "[!]" if lstatus == "UNVERIFIED" else "[X]"
                print(f"\n  {status_icon} Layer {lid:25s} [{lstatus}]")
                
                if layer.get("fail_reasons"):
                    print("    FAIL REASONS:")
                    for reason in layer["fail_reasons"]:
                        print(f"      - {reason}")
                
                print("    CHECKS:")
                for check in layer.get("checks", []):
                    cname = check.get("name")
                    cpass = check.get("pass")
                    cval = check.get("value")
                    
                    # Formatting check status
                    cstatus = "OK" if cpass is True else "!!" if cpass is False else "--"
                    cval_str = ""
                    if isinstance(cval, (float, int)):
                        cval_str = f"({cval:.3f})"
                    elif isinstance(cval, list) and all(isinstance(x, (float, int)) for x in cval):
                        cval_str = f"({', '.join([f'{x:.2f}' for x in cval[:3]])}...)"
                        
                    print(f"      - {cname:25s} [{cstatus}] {cval_str}")
        except Exception as e:
            print(f"  Warning: Could not parse verification report: {e}")
'''

content = content.replace(insertion_marker, verification_logic + "\n    " + insertion_marker)

target.write_text(content, encoding="utf-8")
print("Cleaned up and upgraded verification integration in compare_controls.py")
