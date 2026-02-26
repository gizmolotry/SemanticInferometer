# Belief Transformer - Complete Experimental Workflow

## Overview

This guide walks through the complete experimental pipeline for your thesis.

---

## Phase 1: Analyze Current Results (30 min)

You already have 5 observers on 4,054 real articles. Let's analyze them.

### Run Procrustes Analysis

```bash
python compare_observers.py
```

**This generates:**
- `outputs/attention_analysis.csv` - Attention matrix properties
- `outputs/procrustes_umap.csv` - Pairwise UMAP residuals
- `outputs/procrustes_attention.csv` - Pairwise attention residuals
- `outputs/high_divergence_articles.csv` - Top 50 most divergent articles
- `outputs/residual_heatmap_umap.png` - Visual comparison
- `outputs/residual_heatmap_attention.png` - Visual comparison

**Key metrics to check:**
- Are residual_mean values > 0.01? (variance exists)
- Are some observer pairs more different than others?
- Which articles have highest divergence scores?

---

## Phase 2: Build Control Corpus (15 min)

Generate the three-class control for baseline comparison.

```bash
python make_control_corpus.py
```

**This generates:**
- `data/control_corpus.jsonl` - 1,500 articles (500 each)
  - Class A: Constant (identical text)
  - Class B: Shuffled (same vocab, random order)
  - Class C: Random (vocab soup)

---

## Phase 3: Run Control Experiment (25 min)

Process control corpus through same 5 observers.

```bash
python run_experiments.py --mode control
```

**This generates:**
- `outputs/control_observer_42.pt`
- `outputs/control_observer_43.pt`
- ... (5 files total)

**Expected:** Control residuals should be MUCH smaller than real corpus.

---

## Phase 4: Scrape Temporal Data (2-3 hours, can run overnight)

Gather articles from multiple time periods.

```bash
# Scrape all temporal slices (Oct 2023 - Mar 2024)
python scrape_temporal.py --target 500

# Or scrape specific slice
python scrape_temporal.py --slice Oct_2023 --target 500
```

**This generates:**
- `data/temporal/Oct_2023.jsonl`
- `data/temporal/Nov_2023.jsonl`
- `data/temporal/Dec_2023.jsonl`
- ... (6 files total, ~3,000 articles)

---

## Phase 5: Run Temporal Experiments (3-4 hours)

Process each temporal slice through same observers.

```bash
python run_experiments.py --mode temporal
```

**This generates:**
- `outputs/temporal_Oct_2023_observer_42.pt`
- `outputs/temporal_Oct_2023_observer_43.pt`
- ... (30 files total: 6 slices × 5 observers)

**Goal:** Compare how same observers see different time periods.

---

## Phase 6: Complete Comparison (30 min)

Compare ALL experiments together.

```bash
python run_experiments.py --mode compare
```

**This generates:**
- `outputs/all_procrustes_comparisons.csv` - Residuals across all conditions

**Analysis questions:**
- Real corpus residual > Control residual? (YES = content matters)
- Temporal variance across slices? (observer consistency over time)
- Which time periods show highest divergence?

---

## Quick Run (Everything)

```bash
# Generate control corpus
python make_control_corpus.py

# Run all experiments (real + control + temporal + comparison)
# WARNING: This takes 6-8 hours total
python run_experiments.py --mode all
```

---

## File Structure After Complete Run

```
outputs/
├── real_observer_42.pt              # Real corpus, observer 42
├── real_observer_43.pt              # Real corpus, observer 43
├── ...
├── control_observer_42.pt           # Control corpus, observer 42
├── control_observer_43.pt           # Control corpus, observer 43
├── ...
├── temporal_Oct_2023_observer_42.pt # Oct 2023, observer 42
├── temporal_Nov_2023_observer_42.pt # Nov 2023, observer 42
├── ...
├── attention_analysis.csv           # Attention properties
├── procrustes_umap.csv              # UMAP residuals
├── procrustes_attention.csv         # Attention residuals
├── high_divergence_articles.csv     # Most divergent articles
├── all_procrustes_comparisons.csv   # Complete comparison table
├── residual_heatmap_umap.png        # Visual heatmap
└── residual_heatmap_attention.png   # Visual heatmap
```

---

## Thesis Argument Structure

### Chapter 1: Observer Variance Exists
**Evidence:**
- Procrustes residuals from `procrustes_umap.csv`
- Real corpus residual > 0 (observers differ)

### Chapter 2: Variance is Content-Dependent
**Evidence:**
- Real residual > Control residual
- `all_procrustes_comparisons.csv` shows:
  - Real: residual_mean = X
  - Control constant: residual_mean ≈ 0
  - Control shuffled: residual_mean = Y (Y < X)
  - Control random: residual_mean = Z (Z < X)

### Chapter 3: Temporal Consistency
**Evidence:**
- Same observers across time slices
- Variance in how Oct vs Nov vs Dec are perceived
- `high_divergence_articles.csv` identifies which events are most observer-dependent

### Chapter 4: Implications
- "Bias detection assumes observer-independent stance"
- "We show geometry is fundamentally observer-specific"
- "Therefore: perspective matters in computational stance detection"

---

## Next Steps Priority

**Immediate (tonight):**
1. Run `compare_observers.py` (30 min)
2. Check if variance exists in current data
3. Run `make_control_corpus.py` (5 min)
4. Run control experiment (25 min)

**Tomorrow:**
1. Start temporal scraping (can run overnight)
2. Analyze control vs real residuals
3. Write up initial findings

**This week:**
1. Complete temporal experiments
2. Run full comparison
3. Generate all figures for thesis

---

## Troubleshooting

### "Module not found" errors
```bash
# Make sure you're in the project root
cd D:\belief-transformer\V3
python compare_observers.py
```

### Out of GPU memory
```bash
# Use smaller batch sizes or CPU
# Edit core/complete_pipeline.py line 46:
device = 'cpu'
```

### Scraper rate limited
```bash
# Reduce target per slice
python scrape_temporal.py --target 300
```

---

## Questions?

Email or check:
- `core/procrustes.py` - Math behind residuals
- `core/complete_pipeline.py` - Pipeline architecture
- `run_scraped_data.py` - Original runner

Good luck!
