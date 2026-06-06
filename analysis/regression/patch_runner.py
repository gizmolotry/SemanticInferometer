import os
from pathlib import Path

# Paths
runner_path = Path("..") / "run_experiments.py"

# Read the file
with open(runner_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update load_corpus function
old_load_corpus = 'def load_corpus(corpus_type, limit=None):'
new_load_corpus = """def load_corpus(corpus_type, limit=None):
    # Support for custom corpus paths (Direct Injection)
    if os.path.exists(corpus_type):
        articles = load_articles(Path(corpus_type))
        if limit:
            articles = articles[:limit]
        return articles
"""
if old_load_corpus in content:
    content = content.replace(old_load_corpus, new_load_corpus)

# 2. Remove choices constraint for --corpus
# Note: My previous replace might have already messed this up, so I'll be flexible
import re
content = re.sub(r"parser\.add_argument\(\s*'--corpus',\s*type=str,\s*default='real',\s*choices=\[.*?\],", 
                 "parser.add_argument('--corpus', type=str, default='real',", content, flags=re.DOTALL)

# 3. Update corpus_name variable mapping in main()
# Find the end of args = parser.parse_args() and inject it
old_args_parse = "args = parser.parse_args()"
new_args_parse = """args = parser.parse_args()
    
    # Resolve corpus name for custom paths
    corpus_name = Path(args.corpus).stem if os.path.exists(args.corpus) else args.corpus
"""
if old_args_parse in content:
    content = content.replace(old_args_parse, new_args_parse)

# 4. Global replacement for corpus_name usage in calls
content = content.replace("corpus_name=args.corpus", "corpus_name=corpus_name")
content = content.replace("args.corpus, limit=args.limit", "args.corpus, limit=args.limit")

# 5. Fix batch size and imports
if "import argparse" in content and "import argparse, os" not in content:
    content = content.replace("import argparse", "import argparse, os")

# Ensure batch_size is large for VRAM utilization
content = re.sub(r"batch_size = \d+", "batch_size = 512", content)

# Write the file back
with open(runner_path, "w", encoding="utf-8") as f:
    f.write(content)

print("[OK] run_experiments.py patched successfully.")
