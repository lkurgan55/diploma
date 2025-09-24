# Diploma Project
**Topic:** *Impact of Token Decoding Strategies in Large Language Models on Text-to-SQL Quality*

This repository contains code and experiments for a master’s thesis exploring how decoding strategies (greedy, beam, top-k, top-p, execution-guided beam) affect Text-to-SQL generation quality.

---

## Repository Structure
```
.
├─ decoding/ # Decoding strategies (greedy, beam, top-k, top-p, EG-beam)
├─ outputs/ # Generated predictions, metrics
├─ src/ # Source code: schema utils, prompts, helpers
├─ main.py # Run experiments on a dataset
├─ test_query_model.py # Quick model test on a single prompt
├─ run_metrics.py # Compute metrics for predictions (JSON)
└─ README.md
```
## Results

### Current Results (Model: qwen2.5-3B-Instruct)

| Strategy | Execution Accuracy     | String Match Accuracy | Component Match Accuracy |
|----------|------------------------|-----------------------|--------------------------|
| Greedy   | 174/500 = **0.3480**   | 11/500 = **0.0220**   | 185.4167/500 = **0.3708** |
| Beam     | 196/500 = **0.3920**   | 11/500 = **0.0220**   | 186.7167/500 = **0.3734** |
| Top-k    | 171/500 = **0.3420**   | 11/500 = **0.0220**   | 180.1333/500 = **0.3603** |
| Top-p    | 176/500 = **0.3520**   | 11/500 = **0.0220**   | 180.0500/500 = **0.3601** |
| **EG-Beam** (table/column check) | 206/500 = **0.4120** | 11/500 = **0.0220** | 188.0333/500 = **0.3761** |
