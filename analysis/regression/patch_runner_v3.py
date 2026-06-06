import os
import re

# Fix run_experiments.py load_corpus correctly
runner_path = os.path.join("..", "run_experiments.py")
with open(runner_path, "r", encoding="utf-8") as f:
    content = f.read()

# Unified load_corpus function
new_function = """def load_corpus(corpus_type, limit=None):
    # Support for custom corpus paths
    if os.path.exists(corpus_type):
        articles = load_articles(Path(corpus_type))
        if limit: articles = articles[:limit]
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
    else:
        raise ValueError(f"Unknown corpus type: {corpus_type}")
        
    if limit: articles = articles[:limit]
    return articles
"""

# Replace all existing load_corpus definitions
content = re.sub(r'def load_corpus\(.*?\):.*?return articles', new_function, content, flags=re.DOTALL)

with open(runner_path, "w", encoding="utf-8") as f:
    f.write(content)

print("[OK] load_corpus fixed in run_experiments.py.")
