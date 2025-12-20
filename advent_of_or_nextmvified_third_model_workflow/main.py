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

# Instantitate the Nextmv Cloud sub app. This is only needed to get the sub
# app's assets.
client = cloud.Client(api_key=os.getenv("NEXTMV_API_KEY"))
cloud_sub_app = cloud.Application(client=client, id="gams-portfolio-rebalancing")


class PortfolioOptimizationWorkflow(FlowSpec):
    """
    Decision workflow for portfolio optimization with CVaR risk measure.

    Workflow structure:
        start
          |
          +-- run_objective_1 (parallel) --------+
          |                                      |
          +-- run_objective_2 (parallel)         |
                    |                            |
                    +----------------------------+
                                |
                           pick_best
    """

    @step
    def read_inputs(_) -> dict[str, Any]:
        """
        Load input data and prepare for parallel branches.

        Returns the directory containing input CSV files.
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
        app_id=cloud_sub_app.id,
        options=cloud_opt | {"objective": "objective_1"},
        full_result=True,
    )
    @needs(predecessors=[read_inputs])
    @step
    def run_objective_1() -> None:
        """
        Run two-stage optimization approach (objective_1).

        This branch runs in parallel with the hierarchical multi-objective
        """

        log("\n" + "=" * 60)
        log("BRANCH: Objective 1 - Maximize Profit, Minimize CVaR")
        log("=" * 60)

    @app(
        app_id=cloud_sub_app.id,
        options=cloud_opt | {"objective": "objective_2"},
        full_result=True,
    )
    @needs(predecessors=[read_inputs])
    @step
    def run_objective_2() -> None:
        """
        Run hierarchical multi-objective approach (objective_2).

        This branch runs in parallel with the two-stage approach.
        """

        log("\n" + "=" * 60)
        log("BRANCH: Objective 2 - Single-Stage Weighted Multi-Objective")
        log("=" * 60)

    @needs(predecessors=[run_objective_1, run_objective_2])
    @step
    def pick_best(
        result_1: nextmv.RunResult,
        result_2: nextmv.RunResult,
    ) -> dict[str, Any]:
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
            result = result_1
            selected_approach = "objective_1"

            return {
                "result": result,
                "approach": selected_approach,
                "cvar_1": cvar_1,
                "cvar_2": cvar_2,
            }

        log("\n  Selected: Objective 2 (lower CVaR)")
        result = result_2
        selected_approach = "objective_2"

        return {
            "result": result,
            "approach": selected_approach,
            "cvar_1": cvar_1,
            "cvar_2": cvar_2,
        }

    @needs(predecessors=[pick_best])
    @step
    def write_outputs(pick_best_result: dict[str, Any]) -> None:
        """
        Write selected result outputs to expected locations.
        """

        result = pick_best_result["result"]
        approach = pick_best_result["approach"]
        cvar_1 = pick_best_result["cvar_1"]
        cvar_2 = pick_best_result["cvar_2"]

        result_path = result.output
        result_stats = result.metadata.statistics

        # Get the solution files from the selected result and copy them to the
        # expected location.
        solutions_dir = os.path.join("outputs", "solutions")
        os.makedirs(solutions_dir, exist_ok=True)
        for file_name in os.listdir(result_path):
            full_file_name = os.path.join(result_path, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, solutions_dir)

        # Add selection metadata to statistics to and write them to expected
        # statistics location.
        result_stats["selected_approach"] = approach
        result_stats["objective_1_cvar"] = cvar_1
        result_stats["objective_2_cvar"] = cvar_2
        final_statistics = {"statistics": result_stats}
        stats_dir = os.path.join("outputs", "statistics")
        os.makedirs(stats_dir, exist_ok=True)
        nextmv.write(final_statistics, path=os.path.join(stats_dir, "statistics.json"))

        # Get the assets from the selected result and copy them to the expected
        # location. Right now, only a single asset has all the visuals.
        run_id = result.id
        run_assets = cloud_sub_app.list_assets(run_id)
        assets = []
        for run_asset in run_assets:
            content = cloud_sub_app.download_asset_content(asset=run_asset)
            asset = nextmv.Asset(
                name=run_asset.name,
                content_type=run_asset.content_type,
                visual=run_asset.visual,
                content=content,
            )
            assets.append(asset.to_dict())

        final_assets = {"assets": assets}
        assets_dir = os.path.join("outputs", "assets")
        os.makedirs(assets_dir, exist_ok=True)
        nextmv.write(final_assets, path=os.path.join(assets_dir, "assets.json"))


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
