# Causal Model Combination (CMC) for Time Series with Missing Variables

## Overview

**CMC-TS** is a Python framework for discovering **population-level causal structures** in binary time series datasets where variable measurements vary across instances — a common scenario in healthcare, where data collection is tied to clinical events.

CMC-TS iteratively combines partial observations across datasets to:
1. Identify causal relationships with temporal lag.
2. Group datasets by shared variable structures.
3. Score and rank model combinations.
4. Reconstruct unmeasured (latent) variables using a knowledge base.
5. Refine causal understanding at the population level.

## Key Features

- **Handles heterogeneity**: Works with datasets that have overlapping but non-identical variable sets.
- **Temporal causal discovery**: Supports lagged variable relationships with flexible lag windows.
- **Latent variable reconstruction**: Infers unmeasured variables using knowledge from more complete datasets.
- **Conflict resolution**: Uses statistical techniques to combine and resolve inconsistent relationships across datasets.
- **Statistical rigor**: Applies weighted Fisher’s method and FDR correction for significance testing.
- **Graph-based scoring**: Ranks causal model combinations based on structure, subset completeness, and latent variable influence.
- **Parallel processing**: Accelerates computation with Joblib for scalable inference.

## Project Structure

### Main Script

- `main.py`: Runs the full CMC-Ts pipeline — loading data, iterating through causal discovery and reconstruction stages, and saving outputs.

### Modules

| Module                 | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `CMC_TS.py`            | Main logic for CMC-TS. Manages dataset grouping, causal inference, clustering, scoring, and reconstruction loops. |
| `CMC_SK_prob_pop.py`   | Executes causal discovery using time-lagged relationships across datasets. Outputs edge probabilities and signs. |
| `CMC_combine.py`       | Combines causal results across datasets. Resolves conflicting directions and windows. Applies FDR and weighted Fisher correction. |
| `CMC_score.py`         | Scores causal models based on graph structure (e.g., proper subsets, node roles, latent variable impact). |
| `CMC_Reconstruct.py`   | Reconstructs missing latent variables by simulating their behavior based on known causal relationships and occurrence probabilities. |

## Installation

```bash
pip install pandas numpy matplotlib networkx joblib scipy statsmodels numba pgmpy
```

Python 3.7+ is recommended.

## Usage

```bash
python main.py <dataset_path> <sig_level> <fdr_ind> <min_lag> <max_lag> <lag_interval> <probability_modulator> <max_level> <num_datasets> <effects_to_test> <causes_to_exclude> <linkedvarsgroup>
```

### Example

```bash
python main.py ./example_dataset 0.05 true 1 6 60 0.5 3 10 "V1|V2" "V1|V2|V3|V4" "V1|V2|V3#V5|V6#V10|V11" 
```

## Arguments

| Argument               | Description                                                                          |
|------------------------|--------------------------------------------------------------------------------------|
| `dataset_path`         | Path containing a `/data/` folder with `.csv` files.                                 |
| `sig_level`            | Significance level threshold (e.g., 0.05).                                           |
| `fdr_ind`              | `"true"` to apply FDR correction, `"false"` to skip.                                 |
| `min_lag`, `max_lag`   | Temporal lag range (in seconds).                                                     |
| `lag_interval`         | Step between lags (e.g., 60 for 1-minute bins).                                      |
| `probability_modulator`| Minimum edge probability to keep a causal link.                                      |
| `max_level`            | Maximum number of reconstruction levels.                                             |
| `num_datasets`         | Number of CSV datasets in the `/data/` folder.                                       |
| `effects_to_test`      | The effects you are interested in. Add | between items (e.g., "V1|V2")               |
| `causes_to_exclude`    | The causes you are not interested in testing (e.g., "V1|V2|V3|V4")                   |
| `linkedvarsgroup`      | Indicate which variables are linked (see paper for example) - add # between groups   |
| `linkedvarsgroup`      | Example: "V1|V2|V3#V5|V6#V10|V11"                                                    |


## Input Format

Place `.csv` datasets in `dataset_path/data/`. Each file must contain:
- `pID`: Patient ID
- `timestamp`: Time of measurement
- Other binary columns representing presence/absence of events or variables

## Output

Results are saved to `dataset_path/results/`:

- `CMC_relationships.csv`: Final significant causal relationships with edge probabilities
- `rank_and_score_level_X.csv`: Ranked latent variable reconstructions per iteration
- `CMC_pre-stage_X.pickle`, `CMC_post-stage_X.pickle`: Checkpoint files for recovery or review

## Applications

CMC was developed with healthcare use cases in mind, but its approach generalizes to any domain with:
- Heterogeneous multivariate binary time series
- Missing or context-dependent variable availability
- Need for interpretable causal inference across a population

Examples:
- ICU patient monitoring
- Wearable device signal analysis
- Personalized chronic disease management
- Public health datasets with partial coverage
