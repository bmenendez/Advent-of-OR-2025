import os
import sys

import gamspy as gp
import nextmv
import pandas as pd


def main(
    segments: pd.DataFrame,
    assets: pd.DataFrame,
    covariance_df: pd.DataFrame,
    options: nextmv.Options,
) -> nextmv.Output:
    """
    Portfolio rebalancing optimization model with dual objectives.

    Objective 1: Maximizes net profit (expected profit - transaction costs) subject to:
    - Asset exposure bounds (max increase/decrease limits)
    - Portfolio-level risk weight constraint

    Objective 2: Minimizes portfolio covariance while maintaining minimum profit target subject to:
    - All constraints from Objective 1
    - Minimum profit constraint (target * maximum_profit)

    Args:
        segments: DataFrame with columns: asset, segment_id, exposure, average_profitability,
                  risk_weight, rel_sell_cost, rel_origination_cost
        assets: DataFrame with columns: asset, max_exposure_decrease, max_exposure_increase
        covariance_df: DataFrame with covariance matrix (index and columns are asset names)
        options: nextmv.Options object for solver configuration

    Returns:
        nextmv.Output containing:
            - segment_results: DataFrame with detailed segment-level results
            - asset_results: DataFrame with asset-level aggregated results
            - portfolio_summary: DataFrame with portfolio-level summary metrics
            - model_status: Solver status information
            - covariance: Portfolio covariance (if objective_2 selected)
    """
    m = gp.Container()

    # Sets
    A = gp.Set(
        m,
        description="Set of all assets in the portfolio",
        records=assets["asset"].unique(),
    )
    S = gp.Set(
        m,
        description="Set of segments",
        records=segments["segment_id"],
    )
    AS = gp.Set(
        m,
        domain=[A, S],
        description="Relevant pairs of assets and segments",
        records=segments[["asset", "segment_id"]],
    )

    # Parameters
    average_profitability = gp.Parameter(
        m,
        domain=[A, S],
        description="Expected profitability for asset a, segment s",
        records=segments[["asset", "segment_id", "average_profitability"]],
    )
    risk_weight_limit = gp.Parameter(
        m,
        description="Maximum allowable average risk weight at the portfolio level",
        records=options.risk_weight_limit,
    )
    max_exposure_decrease = gp.Parameter(
        m,
        domain=A,
        description="Lower bound for asset `a` relative exposure",
        records=assets[["asset", "max_exposure_decrease"]],
    )
    max_exposure_increase = gp.Parameter(
        m,
        domain=A,
        description="Upper bound for asset `a` relative exposure",
        records=assets[["asset", "max_exposure_increase"]],
    )
    rel_origination_cost = gp.Parameter(
        m,
        domain=[A, S],
        description="Per unit cost of increasing the exposure of segment s for asset a",
        records=segments[["asset", "segment_id", "rel_origination_cost"]],
    )
    rel_sell_cost = gp.Parameter(
        m,
        domain=[A, S],
        description="Per unit cost of decreasing the exposure of segment s for asset a",
        records=segments[["asset", "segment_id", "rel_sell_cost"]],
    )
    risk_weight = gp.Parameter(
        m,
        domain=[A, S],
        description="Risk weight of segment s for asset a",
        records=segments[["asset", "segment_id", "risk_weight"]],
    )
    exposure = gp.Parameter(
        m,
        domain=[A, S],
        description="Current exposure of segment s for asset a.",
        records=segments[["asset", "segment_id", "exposure"]],
    )
    current_asset_exposure = gp.Parameter(
        m,
        domain=A,
        description="Total exposure of asset a",
    )
    current_asset_exposure[A] = gp.Sum(AS[A, S], exposure[A, S])

    # NEW: Covariance matrix parameter
    # Convert covariance DataFrame to GAMSPy records format
    cov_records = []
    for i in covariance_df.index:
        for j in covariance_df.columns:
            cov_records.append([i, j, covariance_df.loc[i, j]])

    covariance_matrix = gp.Parameter(
        m,
        domain=[A, A],
        description="Covariance matrix of asset returns",
        records=cov_records,
    )

    # Variables
    segment_vars = gp.Variable(
        m,
        domain=[A, S],
        type="Positive",
        description="Multiplier of original exposure for segment, asset in the rebalanced portfolio",
    )
    segment_increase_vars = gp.Variable(
        m,
        domain=[A, S],
        type="Positive",
        description="Increase multiplier of original exposure for segment s, asset a in the rebalanced portfolio",
    )
    segment_decrease_vars = gp.Variable(
        m,
        domain=[A, S],
        type="Positive",
        description="Decrease multiplier of original exposure for segment s, asset a in the rebalanced portfolio",
    )
    new_total_exposure = gp.Variable(
        m,
        description="Total exposure in new rebalanced portfolio",
    )
    portfolio_exposure_vars = gp.Variable(
        m,
        domain=A,
        description="Total exposure in new rebalanced portfolio for asset a",
        type="Positive",
    )

    # Constraints
    # 1. Keep portfolio exposures within allowable limits l_a <= e_a <= u_a
    def_exposure_lo = gp.Equation(
        m,
        domain=A,
        description="Lower bound on exposure change",
    )
    def_exposure_lo[A] = (
        portfolio_exposure_vars[A]
        >= (1 - max_exposure_decrease[A]) * current_asset_exposure[A]
    )

    def_exposure_up = gp.Equation(
        m,
        domain=A,
        description="Upper bound on exposure change",
    )
    def_exposure_up[A] = (
        portfolio_exposure_vars[A]
        <= (1 + max_exposure_increase[A]) * current_asset_exposure[A]
    )

    # 2. Risk Weight Constraint
    def_risk_weight = gp.Equation(
        m,
        description="This constraint keeps the average risk weight at the portfolio-level below the user-defined threshold.",
    )
    def_risk_weight[...] = (
        gp.Sum(AS, risk_weight[AS] * exposure[AS] * segment_vars[AS])
        <= risk_weight_limit * new_total_exposure
    )

    # 3. Segment Relationship Constraint
    segment_relationship = gp.Equation(
        m,
        domain=[A, S],
        description="This establishes the relationship between the main segment variables and their increase/decrease components.",
    )
    segment_relationship[AS] = (
        segment_vars[AS] == 1 + segment_increase_vars[AS] - segment_decrease_vars[AS]
    )

    # 4. Asset Exposure Relationship Constraint
    asset_exposure = gp.Equation(
        m,
        domain=A,
        description="This establishes the relationship between the per segment ratio and the asset level exposure of the rebalanced portfolio.",
    )
    asset_exposure[A] = (
        gp.Sum(AS[A, S], exposure[AS] * segment_vars[AS]) == portfolio_exposure_vars[A]
    )

    # 5. Total Exposure Relationship Constraint
    total_exposure = gp.Equation(
        m,
        description="This establishes the relationship between the per asset exposure and the total exposure of the rebalanced portfolio.",
    )
    total_exposure[...] = gp.Sum(A, portfolio_exposure_vars[A]) == new_total_exposure

    # Objective: Expected net profit
    net_profit = gp.Variable(m, description="Net Profit")
    def_net_profit = gp.Equation(m, description="Definition of net profit")
    def_net_profit[...] = net_profit == gp.Sum(
        AS,
        (
            average_profitability[AS] * segment_vars[AS]
            - rel_origination_cost[AS] * segment_increase_vars[AS]
            - rel_sell_cost[AS] * segment_decrease_vars[AS]
        )
        * exposure[AS],
    )

    # NEW: Covariance objective components
    covariance = gp.Variable(m, description="Portfolio Covariance")
    J = gp.Alias(m, alias_with=A)
    upper = gp.Set(
        m,
        domain=[A, J],
        description="Upper triangular part including diagonal of symmetric covariance matrix",
    )
    lower = gp.Set(
        m,
        domain=[A, J],
        description="Lower triangular part of symmetric covariance matrix",
    )

    # Define upper and lower triangular sets using ordinal indexing
    upper[A, J].where[gp.Ord(A) <= gp.Ord(J)] = True
    lower[A, J].where[gp.Ord(A) > gp.Ord(J)] = True

    # NEW: Risk equation - quadratic covariance calculation
    risk_equation = gp.Equation(m, description="Definition of Portfolio Covariance")
    risk_equation[...] = covariance == gp.Sum(
        upper[A, J],
        portfolio_exposure_vars[A]
        * covariance_matrix[A, J]
        * portfolio_exposure_vars[J],
    ) + gp.Sum(
        lower[A, J],
        portfolio_exposure_vars[A]
        * covariance_matrix[J, A]
        * portfolio_exposure_vars[J],
    )

    # Create model constraints list
    model_constraints = [
        def_exposure_lo,
        def_exposure_up,
        def_risk_weight,
        segment_relationship,
        asset_exposure,
        total_exposure,
        def_net_profit,
    ]

    # Create profit maximization model (LP)
    max_net_profit = gp.Model(
        m,
        problem=gp.Problem.LP,
        sense=gp.Sense.MAX,
        equations=model_constraints,
        objective=net_profit,
    )

    # NEW: Create covariance minimization model (QCP)
    min_covariance = gp.Model(
        m,
        problem=gp.Problem.QCP,
        sense=gp.Sense.MIN,
        equations=model_constraints + [risk_equation],
        objective=covariance,
    )

    # Conditional solve logic based on objective option
    if options.objective == "objective_1":
        # Original behavior: maximize net profit only
        net_profit.lo = 0
        max_net_profit.solve(solver="xpress", output=sys.stdout)
        active_model = max_net_profit
        max_value = None

    elif options.objective == "objective_2":
        # New behavior: two-stage optimization
        # Stage 1: Maximize profit to find maximum
        print("\n=== Stage 1: Maximizing Net Profit ===")
        net_profit.lo = 0
        max_net_profit.solve(solver="xpress", output=sys.stdout)
        max_value = max_net_profit.objective_value
        print(f"Maximum achievable net profit: {max_value:.2f}")

        # Stage 2: Minimize covariance with profit constraint
        print(f"\n=== Stage 2: Minimizing Covariance with Profit Target ===")
        print(f"Profit target: {options.target * 100:.1f}% of maximum = {options.target * max_value:.2f}")
        net_profit.lo = options.target * max_value
        min_covariance.solve(solver="xpress", output=sys.stdout)
        active_model = min_covariance

    else:
        raise ValueError(f"Invalid objective option: {options.objective}. Must be 'objective_1' or 'objective_2'")

    # Generate results
    # Set pandas display options for nice formatting
    pd.set_option("display.max_rows", None)
    pd.options.display.float_format = "{:,.2f}".format

    # Segment-level results
    repCol = gp.Set(
        m,
        records=[
            "old exposure",
            "new exposure",
            "cost incr",
            "cost decr",
            "profit",
            "net profit",
        ],
    )
    repAS = gp.Parameter(m, domain=[A, S, repCol])
    repAS[AS, "old exposure"] = exposure[AS]
    repAS[AS, "new exposure"] = exposure[AS] * segment_vars[AS]
    repAS[AS, "cost incr"] = (
        rel_origination_cost[AS] * segment_increase_vars[AS] * exposure[AS]
    )
    repAS[AS, "cost decr"] = (
        rel_sell_cost[AS] * segment_decrease_vars[AS] * exposure[AS]
    )
    repAS[AS, "profit"] = average_profitability[AS] * segment_vars[AS] * exposure[AS]
    repAS[AS, "net profit"] = (
        repAS[AS, "profit"] - repAS[AS, "cost incr"] - repAS[AS, "cost decr"]
    )

    # Asset-level results
    repA = gp.Parameter(m, domain=[A, repCol])
    repA[A, repCol] = gp.Sum(AS[A, S], repAS[AS, repCol])

    # Portfolio-level summary
    rep = gp.Parameter(m, domain=["*"])
    rep[repCol] = gp.Sum(A, repA[A, repCol])
    rep["risk weight"] = gp.Sum(
        AS[A, S], risk_weight[AS] * exposure[AS] * segment_vars[AS]
    )
    rep["risk weight limit"] = risk_weight_limit * new_total_exposure

    # Add covariance to portfolio summary if using objective_2
    if options.objective == "objective_2":
        rep["covariance"] = covariance
        rep["max possible profit"] = max_value
        rep["target %"] = options.target * 100

    # Convert to DataFrames
    segment_results = repAS.pivot()
    asset_results = repA.pivot()
    portfolio_summary = rep.records

    # Print results based on verbosity
    if options.verbose:
        print("\n=== Segment-Level Results (Detailed) ===")
        print(segment_results)

        print("\n=== Asset-Level Results ===")
        print(asset_results)

    print("\n=== Portfolio Summary ===")
    print(portfolio_summary)

    # NEW: Enhanced statistics with dual objectives
    stats = nextmv.Statistics(
        result=nextmv.ResultStatistics(
            value=net_profit.toValue(),
            duration=active_model.total_solve_time,
            custom={
                "optimized_exposure": new_total_exposure.toValue(),
                "model_status": active_model.status,
                "num_variables": active_model.num_variables,
                "num_constraints": active_model.num_equations,
                "objective_type": options.objective,
                "covariance": covariance.toValue() if options.objective == "objective_2" else None,
                "max_possible_profit": max_value if options.objective == "objective_2" else None,
                "target": options.target if options.objective == "objective_2" else None,
            },
        )
    )
    output = nextmv.Output(
        options=options,
        output_format=nextmv.OutputFormat.MULTI_FILE,
        statistics=stats,
        solution_files=[
            nextmv.csv_solution_file(
                name="segment_results",
                data=segment_results.to_dict(orient="records"),
            ),
            nextmv.csv_solution_file(
                name="asset_results",
                data=asset_results.to_dict(orient="records"),
            ),
            nextmv.csv_solution_file(
                name="portfolio_summary",
                data=portfolio_summary.to_dict(orient="records"),
            ),
        ],
    )

    return output


def get_data(folder: str = "inputs") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load portfolio data from CSV files.

    Args:
        folder: Path to folder containing segments.csv, assets.csv, and covariance.csv (default: "inputs")
                If a relative path is provided, it will be resolved relative to the script's directory.

    Returns:
        Tuple of (segments DataFrame, assets DataFrame, covariance DataFrame)
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # If folder is relative, make it relative to the script directory
    if not os.path.isabs(folder):
        folder = os.path.join(script_dir, folder)

    segments = pd.read_csv(os.path.join(folder, "segments.csv"))
    assets = pd.read_csv(os.path.join(folder, "assets.csv"))
    covariance_df = pd.read_csv(os.path.join(folder, "covariance.csv"), index_col=0)
    return segments, assets, covariance_df


if __name__ == "__main__":
    # Load Nextmv options.
    manifest = nextmv.Manifest.from_yaml(dirpath=".")
    options = manifest.extract_options()

    # Load data
    segments, assets, covariance_df = get_data()

    # Run optimization, verbosity defined by an option.
    output = main(segments, assets, covariance_df, options)
    nextmv.write(output, path="outputs")
