import os
import re
from pathlib import Path

# Unified load_corpus function for run_experiments.py
runner_path = Path("..") / "run_experiments.py"
with open(runner_path, "r", encoding="utf-8") as f:
    content = f.read()

# Locate the start of load_corpus and the end of its typical structure
# We want to replace the whole messy function
new_load_corpus = """def load_corpus(corpus_type, limit=None):
    \"\"\"
    Load corpus based on type - UPDATED with path support!
    \"\"\"
    # Support for custom corpus paths (Direct Injection)
    if os.path.exists(corpus_type):
        articles = load_articles(Path(corpus_type))
        if limit:
            articles = articles[:limit]
        return articles

    if corpus_type == 'control_constant':
        corpus_file = DATA_DIR / 'control_constant.jsonl'
        articles = load_articles(corpus_file)
    elif corpus_type == 'control_shuffled':
        corpus_file = DATA_DIR / 'control_shuffled.jsonl'
        articles = load_articles(corpus_file)
    elif corpus_type == 'control_random':
        corpus_file = DATA_DIR / 'control_random.jsonl'
        articles = load_articles(corpus_file)
    elif corpus_type == 'real':
        corpus_file = DATA_DIR / 'real_corpus.jsonl'
        articles = load_articles(corpus_file)
    elif corpus_type == 'temporal':
        # Temporal is handled separately in main loop usually
        raise ValueError(\"Temporal corpus should be handled via --batch-temporal\")
    else:
        raise ValueError(f\"Unknown corpus type: {corpus_type}\")

    if limit:
        articles = articles[:limit]
    return articles
"""

# Surgical replacement using regex to find the whole function block
# We look for the start and the first occurrence of 'return articles' that closes the function
content = re.sub(r'def load_corpus\(.*?\):.*?return articles', new_load_corpus, content, flags=re.DOTALL)

with open(runner_path, "w", encoding="utf-8") as f:
    f.write(content)

print("[OK] run_experiments.py cleaned and fixed.")
