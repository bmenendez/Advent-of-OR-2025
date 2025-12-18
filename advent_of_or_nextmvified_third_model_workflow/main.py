"""
Portfolio Optimization Decision Workflow

This workflow runs two optimization approaches in parallel:
1. objective_1: Two-stage optimization (Stage 1: max profit -> Stage 2: min CVaR)
2. objective_2: Hierarchical multi-objective (profit - 0.5*CVaR)

The workflow automatically selects the result with the lowest CVaR (risk-averse).
"""

import nextmv
from nextpipe import FlowSpec, step, needs

from model import (
    get_data,
    run_stage1_profit,
    run_stage2_cvar,
    run_objective2,
)


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
    def start(_=None):
        """
        Load input data and prepare for parallel branches.

        Returns data dict to be passed to parallel optimization branches.
        """
        # Load data
        segments, assets, covariance_df = get_data()

        # Load options from manifest
        manifest = nextmv.Manifest.from_yaml(dirpath="advent_of_or_nextmvified_third_model_workflow")
        options = manifest.extract_options()

        return {
            "segments": segments,
            "assets": assets,
            "covariance_df": covariance_df,
            "options": options,
        }

    @needs(predecessors=[start])
    @step
    def run_objective_2(data: dict):
        """
        Run hierarchical multi-objective approach (objective_2).

        This branch runs in parallel with run_stage1_profit_step.
        """
        print("\n" + "=" * 60)
        print("BRANCH: Objective 2 (Hierarchical Multi-Objective)")
        print("=" * 60)

        output = run_objective2(
            segments=data["segments"],
            assets=data["assets"],
            covariance_df=data["covariance_df"],
            options=data["options"],
        )

        return {
            "output": output,
            "approach": "objective_2",
            "cvar": output.statistics.result.custom["CVaR"],
            "net_profit": output.statistics.result.value,
        }

    @needs(predecessors=[start])
    @step
    def run_stage1_profit_step(data: dict):
        """
        Stage 1 of objective_1: Maximize profit to find upper bound.

        This branch runs in parallel with run_objective_2.
        Returns data augmented with maximum_net_profit for Stage 2.
        """
        print("\n" + "=" * 60)
        print("BRANCH: Objective 1 - Stage 1 (Maximize Profit)")
        print("=" * 60)

        result = run_stage1_profit(
            segments=data["segments"],
            assets=data["assets"],
            options=data["options"],
        )

        return {
            **data,
            "maximum_net_profit": result["maximum_net_profit"],
            "stage1_solve_time": result["solve_time"],
        }

    @needs(predecessors=[run_stage1_profit_step])
    @step
    def run_stage2_cvar_step(data: dict):
        """
        Stage 2 of objective_1: Minimize CVaR with profit constraint.

        Depends on run_stage1_profit_step to get the maximum_net_profit value.
        """
        print("\n" + "=" * 60)
        print("BRANCH: Objective 1 - Stage 2 (Minimize CVaR)")
        print("=" * 60)

        output = run_stage2_cvar(
            segments=data["segments"],
            assets=data["assets"],
            covariance_df=data["covariance_df"],
            options=data["options"],
            maximum_net_profit=data["maximum_net_profit"],
        )

        return {
            "output": output,
            "approach": "objective_1",
            "cvar": output.statistics.result.custom["CVaR"],
            "net_profit": output.statistics.result.value,
            "maximum_net_profit": data["maximum_net_profit"],
        }

    @needs(predecessors=[run_objective_2, run_stage2_cvar_step])
    @step
    def pick_best(result_obj2: dict, result_obj1: dict):
        """
        Select the result with the lowest CVaR (risk-averse selection).

        Compares results from both optimization approaches and returns
        the one with lower CVaR.
        """
        print("\n" + "=" * 60)
        print("SELECTING BEST RESULT (Minimum CVaR)")
        print("=" * 60)

        cvar_obj2 = result_obj2["cvar"]
        cvar_obj1 = result_obj1["cvar"]

        print(f"  Objective 1 (Two-Stage) CVaR: {cvar_obj1:.4f}")
        print(f"  Objective 2 (Hierarchical) CVaR: {cvar_obj2:.4f}")

        if cvar_obj1 <= cvar_obj2:
            print(f"\n  Selected: Objective 1 (lower CVaR)")
            selected = result_obj1["output"]
            selected_approach = "objective_1"
        else:
            print(f"\n  Selected: Objective 2 (lower CVaR)")
            selected = result_obj2["output"]
            selected_approach = "objective_2"

        # Add selection metadata to statistics
        selected.statistics.result.custom["selected_approach"] = selected_approach
        selected.statistics.result.custom["objective_1_cvar"] = cvar_obj1
        selected.statistics.result.custom["objective_2_cvar"] = cvar_obj2

        return selected


if __name__ == "__main__":
    # Run workflow locally for testing
    print("=" * 60)
    print("Portfolio Optimization Decision Workflow")
    print("=" * 60)

    # Create workflow instance with name and input (None since we load data in start step)
    flow = PortfolioOptimizationWorkflow("PortfolioOptimization", None)
    flow.get_result(flow.pick_best)

    # Execute workflow
    result = flow.run()

    # Write output
    nextmv.write(result, path="advent_of_or_nextmvified_third_model_workflow/outputs")

    print("\n" + "=" * 60)
    print("Workflow completed successfully!")
    print("=" * 60)
