# Advent of OR - Nextmv-ified (Third Model with Visualizations)

Portfolio rebalancing optimization with CVaR risk measure and custom Plotly visualizations for the Nextmv platform.

## Model Overview

This model combines the CVaR (Conditional Value at Risk) portfolio optimization with interactive visualizations:

**Optimization Features:**
- **Objective 1 (Solution Pool)**: Maximizes net profit, then minimizes CVaR with profit constraint, using XPRESS solution pool to generate multiple alternatives
- **Objective 2 (Hierarchical Multi-Objective)**: Combines profit maximization and CVaR minimization in a hierarchical formulation
- Scenario-based profitability for robust risk assessment
- Cardinality constraints on segment changes

**Visualizations:**
- **Input Analysis Tab:**
  - Correlation Heatmap: Shows asset correlation matrix from covariance data
- **Rebalancing Results Tab:**
  - Treemap: Portfolio composition with % change overlay (size = exposure, color = % change)
  - Butterfly Chart: Diverging bar chart showing exposure changes by asset
  - Waterfall: Net profit contribution by asset

## Pre-requisites

1. Python `>=3.10` installed on your machine.

1. [Install the Nextmv CLI](https://docs.nextmv.io/docs/using-nextmv/setup/install).

1. Install the Nextmv Python SDK with additional dependencies.

   ```bash
   pip install 'nextmv[all]'
   ```

## Run the executable decision model

1. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

1. Run the code.

   ```bash
   $ python main.py

   === Stage 1: Maximizing Net Profit ===
   ... (solver output) ...
   Maximum achievable net profit: XXX.XX

   === Stage 2: Minimizing CVaR with Solution Pool ===
   ... (solver output) ...

   === Portfolio Summary ===
   ... (results) ...

   === Generating Custom Visualizations ===
   Created 2 visualization tabs
   ```

## Run locally with Nextmv

1. Run the `local1.py` script to execute a local run.

   ```bash
   $ python local1.py

   Running CVaR portfolio optimization with visualizations...
   Completed run, generated run ID: local-XXXXXXXX with status succeeded
   ```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `risk_weight_limit` | Maximum average risk weight for the portfolio | 0.5 |
| `objective` | `objective_1` (solution pool) or `objective_2` (hierarchical) | objective_1 |
| `target` | Profit target as fraction of maximum (for CVaR minimization) | 0.9 |
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

## Push to Nextmv Cloud

1. Make sure you have an active Nextmv account and your API key is exported.

   ```bash
   export NEXTMV_API_KEY=<YOUR_NEXTMV_API_KEY>
   ```

1. Create a Nextmv Cloud application (or use existing one).

   ```bash
   $ nextmv app create -n gams-portfolio-rebalancing-viz -a gams-portfolio-rebalancing-viz
   ```

1. Push the code to Nextmv.

   ```bash
   $ nextmv app push -a gams-portfolio-rebalancing-viz
   ```

1. Run remotely from the [Nextmv Console](https://cloud.nextmv.io) or the CLI.

   ```bash
   $ nextmv app run -a gams-portfolio-rebalancing-viz -i inputs
   ```
