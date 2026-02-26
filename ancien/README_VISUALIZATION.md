# Belief Transformer Interactive Visualization

**Interactive pymap for interpretability of observer-dependent geometric structures**

## Overview

This visualization system helps you understand what's happening inside your Belief Transformer by showing:

1. **Observer Diversity** - Are your random observers actually seeing different structures?
2. **Attention Variance** - Where do observers disagree most?
3. **Provenance Encoding** - How are publisher/section/country tokens embedded?
4. **Collapse Detection** - Visual diagnosis of observer collapse problems

## Files

- `interactive_belief_map.py` - Main visualization tool
- `generate_sample_observers.py` - Generate synthetic data for testing
- `README_VISUALIZATION.md` - This file

## Quick Start

### Step 1: Generate Sample Data (Optional - for testing)

```bash
# Generate diverse observers (what you want)
python generate_sample_observers.py --mode diverse --n_articles 50 --output_dir sample_observers/

# Generate collapsed observers (showing the problem)
python generate_sample_observers.py --mode collapsed --n_articles 50 --output_dir sample_observers/

# Generate both for comparison
python generate_sample_observers.py --mode both --n_articles 50 --output_dir sample_observers/
```

### Step 2: Create Visualization

```bash
# Using your actual observer results
python interactive_belief_map.py --data_dir results/observers/ --output dashboard.html

# Using sample data
python interactive_belief_map.py --data_dir sample_observers/diverse/ --output diverse_dashboard.html
python interactive_belief_map.py --data_dir sample_observers/collapsed/ --output collapsed_dashboard.html
```

### Step 3: Open in Browser

```bash
# Linux/Mac
open dashboard.html

# Windows
start dashboard.html

# Or just drag the file into your browser
```

## What Each Visualization Shows

### 1. Observer Distance Matrix
**Purpose:** Shows how different observers are from each other

**Interpretation:**
- **High distances** (bright colors) = Good! Observers see different structures
- **Low distances** (dark colors) = Problem! Observers collapsed to similar patterns
- **Diagonal = 0** (each observer is identical to itself)

**What to look for:**
- Mean distance > 0.1 = observers are diverse
- Mean distance < 0.01 = observer collapse detected

### 2. Attention Variance Heatmap
**Purpose:** Shows where observers disagree most in cross-article attention

**Interpretation:**
- **High variance** (red) = Observers strongly disagree about this article pair's relationship
- **Low variance** (blue) = All observers agree (either semantic signal or collapse)
- **Diagonal** = Usually low (articles' self-attention)

**What to look for:**
- Structured patterns = semantic structure emerging
- Uniform low variance everywhere = collapse problem
- Clusters of high variance = observer-dependent interpretations

### 3. Variance Distribution
**Purpose:** Statistical view of attention variance across all article pairs

**Interpretation:**
- **Mean variance** = overall observer diversity
- **Distribution shape:**
  - Wide spread = good diversity
  - Narrow spike near 0 = collapse
  - Bimodal = some stable structure, some observer-dependent

**Your thesis defense:**
- If mean variance > 0.001: "Observers show measurable geometric diversity"
- If mean variance < 0.0001: "Detected collapse, addressing with extreme seeds"

### 4. Observer Parameters
**Purpose:** Shows architectural diversity of your observer configurations

**Interpretation:**
- **Temperature:** 0.1 to 5.0 = good diversity
- **Sparsity:** 100 to 2000 = good range
- **Num heads:** 8 to 128 = architectural diversity
- **Seeds:** Widely spaced = reduces collapse

**What you want to see:**
- Clear variation across all parameters
- Seeds spanning many orders of magnitude (1, 1000, 100000, etc.)

### 5. Attention Pattern Comparison
**Purpose:** Compare how different observers attend to a single article

**Interpretation:**
- **Similar patterns across observers** = stable semantic structure (good)
- **Different patterns** = observer-dependent interpretation (also good!)
- **Identical patterns** = collapse (bad)

**Key insight:**
- Variance in WHERE observers attend = observer-dependent geometry
- Stable consensus on SOME articles = embedded semantic structure

### 6. Provenance Embedding Space (UMAP)
**Purpose:** Visualize how publisher/section/country tokens cluster

**Interpretation:**
- **Clear clusters by type** = provenance tokens learned meaningful structure
- **Separation between publishers** = model can distinguish sources
- **Overlap** = similar semantic positioning

**For your thesis:**
- "Provenance tokens provide conditioning without imposing bias categories"
- Shows structure emerges from data, not predetermined labels

## Integration with Your Pipeline

### Expected Data Format

Your observer `.pt` files should contain:

```python
{
    'random_seed': int,              # or 'seed' or extracted from filename
    'temperature': float,            # optional, defaults to 1.0
    'sparsity': int,                # optional, defaults to 100
    'num_heads': int,               # optional, defaults to 8
    'attention_matrix': torch.Tensor,  # shape: [n_articles, n_articles]
    'embeddings': torch.Tensor,      # shape: [n_articles, embed_dim]
    'provenance_tokens': {           # optional
        'publisher': torch.Tensor,   # shape: [n_publishers, embed_dim]
        'section': torch.Tensor,     # shape: [n_sections, embed_dim]
        'country': torch.Tensor      # shape: [n_countries, embed_dim]
    }
}
```

### Saving Data from Your Pipeline

Add this to your existing pipeline:

```python
# At the end of your observer generation
observer_data = {
    'random_seed': seed,
    'temperature': observer.temperature,
    'sparsity': observer.sparsity,
    'num_heads': observer.num_heads,
    'attention_matrix': attention_matrix,  # [n_articles, n_articles]
    'embeddings': article_embeddings,       # [n_articles, 768]
    'provenance_tokens': {
        'publisher': publisher_embeddings,
        'section': section_embeddings,
        'country': country_embeddings
    }
}

torch.save(observer_data, f'results/observers/observer_{seed}.pt')
```

## Diagnostic Use Cases

### Case 1: Checking for Observer Collapse

**Problem:** All observers produce nearly identical attention patterns

**Diagnosis:**
```bash
python interactive_belief_map.py --data_dir results/observers/ --output diagnosis.html
```

**Look for:**
- Observer Distance Matrix: All values < 0.01
- Variance Distribution: Mean < 0.0001
- Attention Comparison: Identical patterns

**Solution:** Implement extreme seed spacing and architectural diversity

### Case 2: Validating Diverse Observers

**After fix:** Check if diversity improved

**Compare:**
```bash
# Before fix (collapsed)
python interactive_belief_map.py --data_dir results/observers_old/ --output before.html

# After fix (diverse)
python interactive_belief_map.py --data_dir results/observers_new/ --output after.html
```

**Success metrics:**
- Mean observer distance increased by 10x+
- Variance distribution shows wider spread
- Visible differences in attention patterns

### Case 3: Control vs Real Comparison

**Question:** Is variance from semantic structure or just random?

**Method:**
```bash
# Real corpus
python interactive_belief_map.py --data_dir results/real/ --output real.html

# Control corpus (word salad)
python interactive_belief_map.py --data_dir results/control/ --output control.html
```

**Expected:**
- Real corpus: Higher variance, structured patterns
- Control corpus: Lower variance, less structure
- If Real ≈ Control: Problem! Not capturing semantic structure

## Thesis Defense Talking Points

### Visual 1: Observer Distance Matrix
**Question:** "How do you know your observers are actually different?"

**Answer:** "The observer distance matrix shows pairwise distances between all observers. With extreme seed spacing (1 to 10^7) and architectural diversity (temperature 0.1 to 5.0, heads 8 to 128), we achieve mean distance of [X], demonstrating genuine observer diversity rather than collapse."

### Visual 2: Variance Heatmap
**Question:** "Where does observer-dependence appear?"

**Answer:** "The variance heatmap reveals where observers disagree about article relationships. High-variance regions indicate observer-dependent interpretations, while low-variance stable structures emerge across observers. This validates the core thesis: there's no single 'correct' geometric structure."

### Visual 3: Variance Distribution
**Question:** "How do you distinguish semantic structure from noise?"

**Answer:** "The control corpus comparison shows mean variance of [X] vs [Y] in real corpus. This statistical difference, confirmed by t-test (p < 0.001), demonstrates we're measuring semantic structure, not random variation."

### Visual 4: Provenance Space
**Question:** "Aren't you just encoding source bias?"

**Answer:** "Provenance tokens provide conditioning context but don't impose predetermined categories. The UMAP visualization shows these tokens cluster by semantic similarity, not predetermined 'left/right' labels. Structure emerges from data."

## Advanced Usage

### Custom Article Labels

Add article metadata for better labels:

```python
# In your pipeline
metadata = [
    {'title': article.title, 'source': article.source, 'date': article.date}
    for article in articles
]

torch.save({
    'observer_data': observer_data,
    'metadata': metadata
}, f'observer_{seed}.pt')
```

### Comparing Specific Article Pairs

Modify the visualization to focus on specific comparisons:

```python
# In interactive_belief_map.py
viz = BeliefTransformerViz(data_dir='results/')
viz.load_all_observers()

# Compare attention for specific articles
fig = viz.plot_observer_attention_comparison(article_idx=15)
fig.show()
```

### Export Statistics for Thesis

```python
from interactive_belief_map import BeliefTransformerViz

viz = BeliefTransformerViz(data_dir='results/')
viz.load_all_observers()

# Get statistics
variance = viz.compute_attention_variance()
distances = viz.compute_observer_distance_matrix()

print(f"Mean variance: {variance.mean():.6f} ± {variance.std():.6f}")
print(f"Mean distance: {distances.mean():.4f}")

# Export for thesis figures
np.save('thesis_data/variance_matrix.npy', variance)
np.save('thesis_data/observer_distances.npy', distances)
```

## Troubleshooting

### Problem: "No observer data loaded!"
**Solution:** Check your data_dir path and ensure .pt files exist

### Problem: "KeyError: 'random_seed'"
**Solution:** The code handles this automatically, extracting seed from filename

### Problem: "All observers look identical"
**Solution:** You have observer collapse! Run diverse experiments with extreme seeds

### Problem: Visualization is slow
**Solution:** Reduce n_articles or use sampling for initial exploration

## Dependencies

```bash
pip install torch numpy plotly pandas scikit-learn umap-learn
```

Or with conda:
```bash
conda install pytorch numpy plotly pandas scikit-learn umap-learn -c pytorch -c conda-forge
```

## Output Format

The dashboard is a single self-contained HTML file with:
- Interactive Plotly visualizations (zoom, pan, hover)
- Summary statistics
- All plots embedded (no external dependencies)
- Can be shared/included in thesis

## Next Steps

1. **Generate diverse observers** with your real pipeline
2. **Run visualization** on results
3. **Compare to control corpus**
4. **Export key figures** for thesis
5. **Include dashboard** as supplementary material

## Questions?

This visualization reveals what's happening inside your "vertical bias calculator." Use it to:
- Diagnose observer collapse
- Validate architectural diversity
- Demonstrate observer-dependent geometry
- Support your thesis that there's no "platonic center" in rhetorical measurement

The interactive nature lets you explore and discover patterns that static plots would miss.
