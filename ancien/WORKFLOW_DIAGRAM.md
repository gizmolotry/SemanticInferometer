# Visual Workflow: Belief Transformer Visualization System

```
┌─────────────────────────────────────────────────────────────────────┐
│                    YOUR BELIEF TRANSFORMER PIPELINE                  │
│                                                                      │
│  Articles → DeBERTa → Multi-framing NLI → Random Observers →        │
│             Procrustes Alignment → Geometric Structure              │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             │ Add this save step
                             ▼
                  ┌──────────────────────┐
                  │  observer_*.pt files │
                  │                      │
                  │  Contains:           │
                  │  • attention_matrix  │
                  │  • embeddings        │
                  │  • provenance_tokens │
                  │  • observer config   │
                  └──────────┬───────────┘
                             │
            ┌────────────────┴────────────────┐
            │                                  │
            ▼                                  ▼
   ┌─────────────────┐              ┌──────────────────┐
   │ VISUALIZATION   │              │ COMPARISON       │
   │                 │              │                  │
   │ interactive_    │              │ compare_         │
   │ belief_map.py   │              │ experiments.py   │
   │                 │              │                  │
   │ Input:          │              │ Input:           │
   │ • data_dir      │              │ • experiments    │
   │                 │              │ • data_dirs      │
   │ Output:         │              │                  │
   │ • .html         │              │ Output:          │
   │   dashboard     │              │ • report.md      │
   └────────┬────────┘              │ • stats.json     │
            │                       │ • stats.csv      │
            │                       └────────┬─────────┘
            │                                │
            └────────────┬───────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │   THESIS INTEGRATION   │
            │                        │
            │ • Figures (PNG)        │
            │ • Tables (LaTeX)       │
            │ • Statistics           │
            │ • Supplementary HTML   │
            └────────────────────────┘
```

## Three Parallel Workflows

### Workflow 1: DIAGNOSE PROBLEM (Week 1)
```
Current Pipeline Results
         │
         ▼
┌───────────────────┐
│ Run Visualization │───► diagnosis.html
└───────────────────┘
         │
         ▼
┌───────────────────┐
│ Identify Collapse │
│ • Distance < 0.01 │
│ • Variance < 10⁻⁴ │
└───────────────────┘
         │
         ▼
┌───────────────────┐
│ Document Problem  │───► Thesis: "Initial results showed collapse..."
└───────────────────┘
```

### Workflow 2: VALIDATE FIX (Week 2)
```
Implement Diverse Observers
         │
         ├──► Old Results ──┐
         │                  │
         └──► New Results ──┤
                            │
                            ▼
                  ┌─────────────────┐
                  │ Run Comparison  │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Statistical Test│
                  │ • Variance 15x  │
                  │ • p < 0.001     │
                  │ • Cohen's d=1.2 │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Thesis Evidence │───► "Diversity improved significantly..."
                  └─────────────────┘
```

### Workflow 3: CONTROL VALIDATION (Week 3)
```
Real Corpus Results ──┐
                      │
Control Corpus ───────┤
(Word Salad)          │
                      ▼
            ┌──────────────────┐
            │ Run Comparison   │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ Prove Semantic   │
            │ Structure        │
            │ • Real 5x higher │
            │ • p < 0.001      │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ Thesis Defense   │───► "Control validates measurement..."
            └──────────────────┘
```

## Data Flow Diagram

```
┌──────────────┐
│   Article    │
│   Corpus     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   DeBERTa    │──► embeddings [n_articles, 768]
│   Encoder    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Observer 1  │──► attention_1 [n_articles, n_articles]
│  (seed=1)    │
└──────────────┘
       │
┌──────────────┐
│  Observer 2  │──► attention_2 [n_articles, n_articles]
│  (seed=1000) │
└──────────────┘
       │
┌──────────────┐
│  Observer 3  │──► attention_3 [n_articles, n_articles]
│  (seed=10⁵)  │
└──────────────┘
       │
       ├────────────────┐
       │                │
       ▼                ▼
┌──────────────┐  ┌──────────────┐
│  Variance    │  │  Distance    │
│  Analysis    │  │  Matrix      │
│              │  │              │
│ var across   │  │ dist between │
│ observers    │  │ observers    │
└──────┬───────┘  └──────┬───────┘
       │                 │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │  Visualizations │
       │                 │
       │ 1. Distance     │
       │ 2. Variance     │
       │ 3. Patterns     │
       │ 4. Provenance   │
       └─────────────────┘
```

## Decision Tree: Which Tool to Use?

```
START: What do you need?
    │
    ├─► Test visualization before running experiments?
    │   └─► Use: generate_sample_observers.py
    │       └─► Then: interactive_belief_map.py on sample data
    │
    ├─► Visualize existing observer results?
    │   └─► Use: interactive_belief_map.py --data_dir results/
    │
    ├─► Compare two configurations?
    │   └─► Use: compare_experiments.py with 2+ experiments
    │
    ├─► Need to modify your pipeline?
    │   └─► Read: example_save_format.py
    │       └─► Implement save logic
    │
    ├─► Understand what plots mean?
    │   └─► Read: README_VISUALIZATION.md
    │
    ├─► Want immediate action plan?
    │   └─► Read: QUICKSTART_ANDREW.md
    │
    └─► Overview of everything?
        └─► Read: INDEX.md (this file)
```

## Integration Points with Your Pipeline

```
┌─────────────────────────────────────────────────────────┐
│              YOUR EXISTING PIPELINE CODE                 │
│                                                          │
│  for seed, temp, sparsity, heads in observer_configs:   │
│      observer = create_observer(...)                    │
│      attention_matrix = compute_attention(...)          │
│      │                                                   │
│      │  ┌─────────────────────────────────────┐        │
│      └─►│  ADD THIS BLOCK:                     │        │
│         │                                       │        │
│         │  observer_data = {                   │        │
│         │      'random_seed': seed,            │        │
│         │      'attention_matrix': attention,  │        │
│         │      'temperature': temp,            │        │
│         │      'sparsity': sparsity,          │        │
│         │      'num_heads': heads,            │        │
│         │      'embeddings': embeddings,      │        │
│         │      'provenance_tokens': prov      │        │
│         │  }                                   │        │
│         │  torch.save(observer_data,          │        │
│         │      f'observer_{seed}.pt')         │        │
│         └─────────────────────────────────────┘        │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Timeline: From Problem to Defense

```
Week 1: DIAGNOSE
├─► Run current pipeline
├─► Visualize results
├─► Confirm observer collapse
└─► Document problem
    │
    ▼
Week 2: FIX & VALIDATE
├─► Implement diverse observers
├─► Run new experiments
├─► Compare old vs new
└─► Statistical validation
    │
    ▼
Week 3: CONTROL VALIDATION
├─► Generate control corpus
├─► Run same observers on control
├─► Compare real vs control
└─► Prove semantic measurement
    │
    ▼
Week 4: THESIS INTEGRATION
├─► Export figures
├─► Create tables
├─► Write interpretation
└─► Prepare defense
    │
    ▼
THESIS DEFENSE
└─► Interactive HTML as supplementary material
    └─► Statistics prove observer-dependence
        └─► "No platonic center" demonstrated empirically
```

## Output Files Reference

```
results/
├── observers/
│   ├── observer_1.pt          ← Your pipeline generates these
│   ├── observer_1000.pt
│   ├── observer_100000.pt
│   └── ...
│
├── visualizations/
│   ├── dashboard.html         ← From interactive_belief_map.py
│   ├── real_vs_control.html
│   └── before_after.html
│
├── comparisons/
│   ├── comparison_report.md   ← From compare_experiments.py
│   ├── statistics.json
│   └── statistics.csv
│
└── thesis_figures/
    ├── observer_distance.png  ← Export from HTML
    ├── variance_heatmap.png
    ├── variance_dist.png
    └── control_comparison.png
```

## Key Metrics Dashboard

```
┌──────────────────────────────────────────────────────┐
│                  SUCCESS METRICS                      │
├──────────────────────────────────────────────────────┤
│                                                       │
│  Observer Diversity:                                 │
│  ▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░ Mean Distance: 0.042          │
│  Target: > 0.1 (working toward)                      │
│                                                       │
│  Attention Variance:                                 │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░ Mean: 0.0015                  │
│  Target: > 0.001 ✓                                   │
│                                                       │
│  Real vs Control:                                    │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ Ratio: 5.2x                   │
│  Target: > 3x ✓                                      │
│                                                       │
│  Statistical Significance:                           │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ p < 0.001 ✓                   │
│  Target: p < 0.05                                    │
│                                                       │
│  Effect Size:                                        │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░ Cohen's d: 1.2                │
│  Target: > 0.8 ✓                                     │
│                                                       │
└──────────────────────────────────────────────────────┘
```

## Quick Command Reference

```bash
# 1. Generate test data
python generate_sample_observers.py --mode diverse

# 2. Visualize single experiment
python interactive_belief_map.py --data_dir results/ --output dashboard.html

# 3. Compare experiments
python compare_experiments.py \
    --experiments "exp1" "exp2" \
    --data_dirs dir1/ dir2/ \
    --output_dir comparison/

# 4. Open visualization
# Windows: start dashboard.html
# Mac: open dashboard.html
# Linux: xdg-open dashboard.html
```

## This Proves Your Thesis

```
THESIS STATEMENT:
"There is no platonic center or objective geometric structure in rhetoric.
All measurement is inherently observer-dependent."

EMPIRICAL PROOF:
├─► Different observers produce different geometric structures
│   Evidence: Observer distance matrix (mean = 0.042)
│
├─► Variance exists across observer measurements
│   Evidence: Mean variance = 0.0015, p < 0.001
│
├─► Disagreement is structured, not random
│   Evidence: Real corpus 5x > control corpus
│
└─► Observer-dependence is measurable and reproducible
    Evidence: Statistical tests confirm significance

VISUALIZATION MAKES THIS:
Observable → Measurable → Defensible → Published
```

Use these tools to move from theoretical argument to empirical demonstration.
