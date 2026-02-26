from pathlib import Path

target = Path("../run_full_experiment_suite.py")
content = target.read_text(encoding="utf-8")

# Fix the broken print statement in VERIFICATION HARNESS
content = content.replace('print("\n" + "="*80)', 'print("\\n" + "="*80)') # Just in case it was already correct but escaped
# Actually the output showed: print("\n" + "="*80) but the error was unterminated string
# Looking at the output:
# print("
# " + "="*80)
# It seems there is a literal newline inside the quotes.

content = content.replace('print("\n"', 'print("\\n"')

target.write_text(content, encoding="utf-8")
print("Fixed harness print syntax in run_full_experiment_suite.py")
