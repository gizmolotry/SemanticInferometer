import json
import random
import os
from pathlib import Path

def generate_controls(input_path, output_dir):
    with open(input_path, 'r', encoding='utf-8') as f:
        articles = [json.loads(line) for line in f]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Constant Control: Every article has the same text
    constant_path = Path(output_dir) / "synthetic_constant.jsonl"
    first_text = articles[0].get('content', articles[0].get('body', ''))
    with open(constant_path, 'w', encoding='utf-8') as f:
        for a in articles:
            a_copy = a.copy()
            if 'content' in a_copy: a_copy['content'] = first_text
            if 'body' in a_copy: a_copy['body'] = first_text
            f.write(json.dumps(a_copy) + '\n')
    
    # 2. Shuffled Control: Words in each article are shuffled
    shuffled_path = Path(output_dir) / "synthetic_shuffled.jsonl"
    with open(shuffled_path, 'w', encoding='utf-8') as f:
        for a in articles:
            a_copy = a.copy()
            text_key = 'content' if 'content' in a else 'body'
            text = a.get(text_key, '')
            words = text.split()
            random.shuffle(words)
            a_copy[text_key] = ' '.join(words)
            f.write(json.dumps(a_copy) + '\n')
            
    # 3. Random Control: Random words from a dictionary
    random_path = Path(output_dir) / "synthetic_random.jsonl"
    # Simple word list for random generation
    vocab = "the of and a to in is you that it he was for on are as with his they at be this from I have or by one had not but what all were when we there can an which their said if do each about how up out them then she many some so these would other into has more her two like him see time could no make than first been its who now people my made over did down only way find use may water long little very after words called".split()
    
    with open(random_path, 'w', encoding='utf-8') as f:
        for a in articles:
            a_copy = a.copy()
            text_key = 'content' if 'content' in a else 'body'
            text = a.get(text_key, '')
            word_count = len(text.split())
            random_words = [random.choice(vocab) for _ in range(word_count)]
            a_copy[text_key] = ' '.join(random_words)
            f.write(json.dumps(a_copy) + '\n')

    print(f"[OK] Generated synthetic controls in {output_dir}")
    return {
        "constant": str(constant_path),
        "shuffled": str(shuffled_path),
        "random": str(random_path)
    }

if __name__ == "__main__":
    generate_controls("../sythgen/high_quality_articles.jsonl", "outputs/synthetic_controls")
