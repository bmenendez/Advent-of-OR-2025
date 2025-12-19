# Advent of OR - Nextmv-ified (Third Model with Decision Workflow)

Portfolio rebalancing optimization with CVaR risk measure using a **Nextmv
Decision Workflow** to orchestrate parallel optimization approaches.

## Decision Workflow Overview

This implementation uses [Nextpipe](https://nextpipe.docs.nextmv.io/en/latest/)
to orchestrate multiple optimization strategies in a single workflow:

```text
                    ┌─────────────────┐
                    │   start (input) │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │  run_objective_2│           │run_stage1_profit│
    │  (single-stage) │           │  (max profit)   │
    └────────┬────────┘           └────────┬────────┘
             │                             │
             │                             ▼
             │                   ┌─────────────────────┐
             │                   │ run_stage2_cvar     │
             │                   │ (min CVaR w/ floor) │
             │                   └────────┬────────────┘
             │                             │
             └──────────────┬──────────────┘
                            ▼
                  ┌─────────────────┐
                  │    pick_best    │
                  │ (min CVaR wins) │
                  └─────────────────┘
```

**Workflow Features:**

- **Parallel Execution**: Objective 1 (two-stage) and Objective 2
  (hierarchical) run in parallel
- **Sequential Stages**: For Objective 1, Stage 1 (maximize profit) must
  complete before Stage 2 (minimize CVaR with profit floor)
- **Automatic Selection**: The workflow automatically selects the result with
  the lowest CVaR (risk-averse)

## Model Overview

**Optimization Approaches:**

- **Objective 1 (Two-Stage Solution Pool)**:
  - Stage 1: Maximizes net profit to find upper bound
  - Stage 2: Minimizes CVaR with profit constraint (target × max_profit), using
    XPRESS solution pool
- **Objective 2 (Hierarchical Multi-Objective)**: Combines profit maximization
  and CVaR minimization in a weighted formulation (profit - 0.5 × CVaR)

**Visualizations:**

- **Input Analysis Tab:**
  - Correlation Heatmap: Shows asset correlation matrix from covariance data
- **Rebalancing Results Tab:**
  - Treemap: Portfolio composition with % change overlay (size = exposure,
    color = % change)
  - Butterfly Chart: Diverging bar chart showing exposure changes by asset
  - Waterfall: Net profit contribution by asset

## Pre-requisites

1. Python `>=3.10` installed on your machine.

1. [Install the Nextmv CLI](https://docs.nextmv.io/docs/using-nextmv/setup/install).

1. Install the Nextmv Python SDK with additional dependencies.

   ```bash
   pip install 'nextmv[all]'
   ```

## Run the decision workflow

1. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

1. Run the workflow.

   ```bash
   $ python main.py

   ============================================================
   Portfolio Optimization Decision Workflow
   ============================================================

   ============================================================
   BRANCH: Objective 2 (Hierarchical Multi-Objective)
   ============================================================
   ... (solver output) ...

   ============================================================
   BRANCH: Objective 1 - Stage 1 (Maximize Profit)
   ============================================================
   ... (solver output) ...

   ============================================================
   BRANCH: Objective 1 - Stage 2 (Minimize CVaR)
   ============================================================
   ... (solver output) ...

   ============================================================
   SELECTING BEST RESULT (Minimum CVaR)
   ============================================================
     Objective 1 (Two-Stage) CVaR: X.XXXX
     Objective 2 (Hierarchical) CVaR: X.XXXX

     Selected: Objective X (lower CVaR)

   ============================================================
   Workflow completed successfully!
   ============================================================
   ```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `risk_weight_limit` | Maximum average risk weight for the portfolio | 0.5 |
| `target` | Profit target as fraction of maximum (for CVaR minimization in Objective 1) | 0.9 |
| `confidence_level` | Confidence level for CVaR calculation | 0.95 |
| `total_num_decrease` | Max fraction of segments that can decrease | 0.3 |
| `total_num_increase` | Max fraction of segments that can increase | 0.4 |
| `total_num_change` | Max fraction of segments that can change | 0.5 |
| `verbose` | Print detailed segment/asset-level results | false |
| `export_latex` | Export model formulation to LaTeX | false |

## Input Files

- `segments.csv`: Asset segments with scenario-based profitabilities (`average_profitability_1`, `average_profitability_2`, etc.)
- `assets.csv`: Asset-level exposure bounds
- `covariance.csv`: Covariance matrix for correlation visualization

## Output

The workflow returns the result from the approach with the **lowest CVaR** (most risk-averse). The output statistics include:

- `selected_approach`: Which approach was selected (`objective_1` or `objective_2`)
- `objective_1_cvar`: CVaR value from the two-stage approach
- `objective_2_cvar`: CVaR value from the hierarchical approach

## Project Structure

```
advent_of_or_nextmvified_third_model_workflow/
├── main.py              # Workflow definition (nextpipe FlowSpec)
├── model.py             # Optimization model with 3 entry points
├── visualizations.py    # Plotly visualization generators
├── app.yaml             # Nextmv app configuration
├── requirements.txt     # Python dependencies
└── inputs/
    ├── assets.csv
    ├── segments.csv
    └── covariance.csv
```

## Push to Nextmv Cloud

1. Make sure you have an active Nextmv account and your API key is exported.

   ```bash
   export NEXTMV_API_KEY=<YOUR_NEXTMV_API_KEY>
   ```

1. Create a Nextmv Cloud application (or use existing one).

   ```bash
   nextmv app create -n gams-portfolio-workflow -a gams-portfolio-workflow
   ```

1. Push the code to Nextmv.

   ```bash
   nextmv app push -a gams-portfolio-workflow
   ```

1. Run remotely from the [Nextmv Console](https://cloud.nextmv.io) or the CLI.

   ```bash
   nextmv app run -a gams-portfolio-workflow -i inputs
   ```
