"""
Portfolio Optimization Decision Workflow

This workflow runs two optimization approaches in parallel:
1. objective_1: Two-stage optimization (Stage 1: max profit -> Stage 2: min CVaR)
2. objective_2: Hierarchical multi-objective (profit - 0.5*CVaR)

The workflow automatically selects the result with the lowest CVaR (risk-averse).
"""

import os
import shutil
from typing import Any

import nextmv
import pandas as pd
from nextmv import cloud
from nextpipe import FlowSpec, app, log, needs, step

# Load options from manifest
manifest = nextmv.Manifest.from_yaml(dirpath=".")
options = manifest.extract_options()
cloud_opt = options.to_dict_cloud()


class PortfolioOptimizationWorkflow(FlowSpec):
    """
    Decision workflow for portfolio optimization with CVaR risk measure.

    Workflow structure:
        start
          |
          +-- run_objective_2 (parallel) --------+
          |                                      |
          +-- run_stage1_profit_step (parallel)  |
                    |                            |
              run_stage2_cvar_step               |
                    |                            |
                    +----------------------------+
                                |
                           pick_best
    """

    @step
    def start(_) -> dict[str, Any]:
        """
        Load input data and prepare for parallel branches.

        Returns data dict to be passed to parallel optimization branches.
        """
        # Load data
        segments, assets, covariance_df = get_data()

        # Save to inputs.
        inputs_dir = "workflow_inputs"
        os.makedirs(inputs_dir, exist_ok=True)
        segments.to_csv(os.path.join(inputs_dir, "segments.csv"), index=False)
        assets.to_csv(os.path.join(inputs_dir, "assets.csv"), index=False)
        covariance_df.to_csv(os.path.join(inputs_dir, "covariance.csv"))

        return inputs_dir

    @app(
        app_id="gams-portfolio-rebalancing",
        options=cloud_opt | {"objective": "objective_1"},
        full_result=True,
    )
    @needs(predecessors=[start])
    @step
    def run_objective_1() -> None:
        """
        Run two-stage optimization approach (objective_1).
        """

        log("\n" + "=" * 60)
        log("BRANCH: Objective 1 - Maximize Profit, Minimize CVaR")
        log("=" * 60)

    @app(
        app_id="gams-portfolio-rebalancing",
        options=cloud_opt | {"objective": "objective_2"},
        full_result=True,
    )
    @needs(predecessors=[start])
    @step
    def run_objective_2() -> None:
        """
        Run hierarchical multi-objective approach (objective_2).

        This branch runs in parallel with run_stage1_profit_step.
        """
        log("\n" + "=" * 60)
        log("BRANCH: Objective 2 - Single-Stage Weighted Multi-Objective")
        log("=" * 60)

    @needs(predecessors=[run_objective_1, run_objective_2])
    @step
    def pick_best(result_1: nextmv.RunResult, result_2: nextmv.RunResult):
        """
        Select the result with the lowest CVaR (risk-averse selection).

        Compares results from both optimization approaches and returns
        the one with lower CVaR.
        """
        log("\n" + "=" * 60)
        log("SELECTING BEST RESULT (Minimum CVaR)")
        log("=" * 60)

        cvar_1 = result_1.metadata.statistics["result"]["custom"]["CVaR"]
        cvar_2 = result_2.metadata.statistics["result"]["custom"]["CVaR"]

        log(f"  Objective 1 (Two-Stage) CVaR: {cvar_1:.4f}")
        log(f"  Objective 2 (Hierarchical) CVaR: {cvar_2:.4f}")

        if cvar_1 <= cvar_2:
            log("\n  Selected: Objective 1 (lower CVaR)")
            result_path = result_1.output
            result_stats = result_1.metadata.statistics
            selected_approach = "objective_1"
        else:
            log("\n  Selected: Objective 2 (lower CVaR)")
            result_path = result_2.output
            result_stats = result_2.metadata.statistics
            selected_approach = "objective_2"

        # Simply copy the files from the given directory to the expected output
        # directory.
        solutions_dir = os.path.join("outputs", "solutions")
        os.makedirs(solutions_dir, exist_ok=True)
        for file_name in os.listdir(result_path):
            full_file_name = os.path.join(result_path, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, solutions_dir)

        # Add selection metadata to statistics to and write them to final location.
        result_stats["selected_approach"] = selected_approach
        result_stats["objective_1_cvar"] = cvar_1
        result_stats["objective_2_cvar"] = cvar_2
        final_statistics = {"statistics": result_stats}
        stats_dir = os.path.join("outputs", "statistics")
        os.makedirs(stats_dir, exist_ok=True)
        nextmv.write(
            final_statistics,
            path=os.path.join(stats_dir, "statistics.json"),
        )


def get_data(folder: str = "inputs") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load portfolio data from CSV files.

    Args:
        folder: Path to folder containing segments.csv, assets.csv, and covariance.csv

    Returns:
        Tuple of (segments DataFrame, assets DataFrame, covariance DataFrame)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if not os.path.isabs(folder):
        folder = os.path.join(script_dir, folder)

    segments = pd.read_csv(os.path.join(folder, "segments.csv"))
    assets = pd.read_csv(os.path.join(folder, "assets.csv"))
    covariance_df = pd.read_csv(os.path.join(folder, "covariance.csv"), index_col=0)
    return segments, assets, covariance_df


if __name__ == "__main__":
    # Run workflow locally for testing
    log("=" * 60)
    log("Portfolio Optimization Decision Workflow")
    log("=" * 60)

    # Create workflow instance with name and input (None since we load data in
    # start step)
    client = cloud.Client(api_key=os.getenv("NEXTMV_API_KEY"))
    flow = PortfolioOptimizationWorkflow(
        name="PortfolioOptimization",
        input=None,
        client=client,
    )

    # Execute workflow. The last step of the flow already prepares the output
    # in the requested directory, so no need to do anything here anymore.
    flow.run()

    log("\n" + "=" * 60)
    log("Workflow completed successfully!")
    log("=" * 60)
