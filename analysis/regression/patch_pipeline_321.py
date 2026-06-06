import re
from pathlib import Path

def patch_pipeline_321_regex():
    path = Path('../core/complete_pipeline.py')
    if not path.exists():
        print("Path not found:", path)
        return
    
    content = path.read_text(encoding='utf-8')

    # Regex to find the loop starting with 'for i, (w, s) in enumerate(zip(walker_work_integrals, walker_states)):'
    # and ending before 'phantom_verdicts.append({'
    pattern = r'for i, \(w, s\) in enumerate\(zip\(walker_work_integrals, walker_states\)\):.*?(?=phantom_verdicts\.append\(\{)'
    
    new_loop_body = r'''for i, (w, s) in enumerate(zip(walker_work_integrals, walker_states)):
                            ratio = float(phantom_ratio[i]) if i < len(phantom_ratio) else 1.0
                            
                            # 3+2+1 OVERHAUL: Family B (Topological Breaks) have priority
                            if s == "broken":
                                verdict = "RUPTURE" # Map to Rupture for HUD, UI handles Laser
                            elif s == "trapped":
                                verdict = "TAUTOLOGY" # Map to Tautology for HUD, UI handles Stall
                            # Family A (Spectral Paths) based on Ratio
                            elif ratio > 2.0: # ratio >> 1
                                verdict = "PHANTOM"
                            elif ratio < 0.5: # ratio << 1
                                verdict = "TAUTOLOGY"
                            else: # ratio approx 1
                                verdict = "HONEST"
                            '''
    
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, new_render_func if 'new_render_func' in locals() else new_loop_body, content, flags=re.DOTALL)
        print("Updated phantom_verdicts loop via regex.")
    else:
        print("Regex failed to match.")

    path.write_text(content, encoding='utf-8')

if __name__ == "__main__":
    patch_pipeline_321_regex()
