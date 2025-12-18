"""
Portfolio rebalancing optimization model with CVaR risk measure.

This module provides three entry points for the workflow:
- run_stage1_profit: Maximize profit only (Stage 1 of objective_1)
- run_stage2_cvar: Minimize CVaR with profit constraint (Stage 2 of objective_1)
- run_objective2: Hierarchical multi-objective optimization
"""

import logging
import os
import sys

import gamspy as gp
import nextmv
import pandas as pd

from visualizations import create_all_visualizations


def _create_base_model(
    segments: pd.DataFrame,
    assets: pd.DataFrame,
    options: nextmv.Options,
):
    """
    Create the base GAMS model with sets, parameters, variables, and constraints.

    Returns a dict containing the model container and all necessary components.
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
    fix_origination_cost = gp.Parameter(
        m,
        domain=[A, S],
        description="Fix cost of increasing the exposure of segment s for asset a",
        records=segments[["asset", "segment_id", "fix_origination_cost"]],
    )
    rel_origination_cost = gp.Parameter(
        m,
        domain=[A, S],
        description="Per unit cost of increasing the exposure of segment s for asset a",
        records=segments[["asset", "segment_id", "rel_origination_cost"]],
    )
    fix_sell_cost = gp.Parameter(
        m,
        domain=[A, S],
        description="Fix cost of decreasing the exposure of segment s for asset a",
        records=segments[["asset", "segment_id", "fix_sell_cost"]],
    )
    rel_sell_cost = gp.Parameter(
        m,
        domain=[A, S],
        description="Per unit cost of decreasing the exposure of segment s for asset a",
        records=segments[["asset", "segment_id", "rel_sell_cost"]],
    )
    total_num_decrease = gp.Parameter(
        m,
        description="Percentage of asset segments that can shrink",
        records=options.total_num_decrease,
    )
    total_num_increase = gp.Parameter(
        m,
        description="Percentage of asset segments that can grow",
        records=options.total_num_increase,
    )
    total_num_change = gp.Parameter(
        m,
        description="Percentage of asset segments that can change",
        records=options.total_num_change,
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

    # Scenario-based profitability
    scenarios = int(segments.columns.str.startswith("average_profitability").sum())
    columns = [f"average_profitability_{idx}" for idx in range(1, scenarios + 1)]
    scen = gp.Set(m, description="Scenarios for CVaR risk measure", records=columns)

    df = segments[["asset", "segment_id"] + columns]
    stacked_df = df.melt(
        id_vars=["asset", "segment_id"],
        value_vars=columns,
        var_name="scen",
        value_name="value",
    )
    average_profitability_scen = gp.Parameter(
        m,
        domain=[A, S, scen],
        description="Expected profitability for asset a, segment s in scenario",
    )
    average_profitability_scen.setRecords(stacked_df)

    average_profitability = gp.Parameter(
        m,
        domain=[A, S],
        description="Expected profitability for asset a, segment s (average across scenarios)",
    )
    average_profitability[AS] = gp.Sum(scen, average_profitability_scen[AS, scen]) / gp.Card(scen)

    # Variables
    segment_vars = gp.Variable(
        m,
        domain=[A, S],
        type="Positive",
        description="Multiplier of original exposure for segment, asset in the rebalanced portfolio",
    )
    segment_increase_bvars = gp.Variable(
        m,
        domain=[A, S],
        type="Binary",
        description="Indicator for increase of original exposure for segment s, asset a",
    )
    segment_increase_vars = gp.Variable(
        m,
        domain=[A, S],
        type="Positive",
        description="Increase multiplier of original exposure for segment s, asset a",
    )
    segment_decrease_bvars = gp.Variable(
        m,
        domain=[A, S],
        type="Binary",
        description="Indicator for decrease of original exposure for segment s, asset a",
    )
    segment_decrease_vars = gp.Variable(
        m,
        domain=[A, S],
        type="Positive",
        description="Decrease multiplier of original exposure for segment s, asset a",
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
    # 1. Keep portfolio exposures within allowable limits
    def_exposure_lo = gp.Equation(
        m,
        domain=A,
        description="Lower bound on exposure change",
    )
    def_exposure_lo[A] = (
        portfolio_exposure_vars[A]
        >= (1 - max_exposure_decrease[A]) * current_asset_exposure[A]
    )

    def_exposure_fx_lo = gp.Equation(
        m,
        domain=[A, S],
        description="Definition of indicator for exposure decrease",
    )
    def_exposure_fx_lo[AS[A, S]] = (
        segment_decrease_vars[AS]
        <= (1 - max_exposure_decrease[A]) * current_asset_exposure[A] * segment_decrease_bvars[AS]
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

    def_exposure_fx_up = gp.Equation(
        m,
        domain=[A, S],
        description="Definition of indicator for exposure increase",
    )
    def_exposure_fx_up[AS[A, S]] = (
        segment_increase_vars[AS]
        <= (1 + max_exposure_increase[A]) * current_asset_exposure[A] * segment_increase_bvars[AS]
    )

    # Cardinality constraints
    def_num_decrease = gp.Equation(
        m,
        description="Limit number of asset segments that shrink in new portfolio",
    )
    def_num_decrease[...] = gp.Sum(AS, segment_decrease_bvars[AS]) <= total_num_decrease * gp.Card(AS)

    def_num_increase = gp.Equation(
        m,
        description="Limit number of asset segments that grow in new portfolio",
    )
    def_num_increase[...] = gp.Sum(AS, segment_increase_bvars[AS]) <= total_num_increase * gp.Card(AS)

    def_num_changes = gp.Equation(
        m,
        description="Limit number of asset segments that change in new portfolio",
    )
    def_num_changes[...] = (
        gp.Sum(AS, segment_decrease_bvars[AS] + segment_increase_bvars[AS])
        <= total_num_change * gp.Card(AS)
    )

    # 2. Risk Weight Constraint
    def_risk_weight = gp.Equation(
        m,
        description="Keep average risk weight at portfolio-level below threshold",
    )
    def_risk_weight[...] = (
        gp.Sum(AS, risk_weight[AS] * exposure[AS] * segment_vars[AS])
        <= risk_weight_limit * new_total_exposure
    )

    # 3. Segment Relationship Constraint
    segment_relationship = gp.Equation(
        m,
        domain=[A, S],
        description="Relationship between segment variables and increase/decrease components",
    )
    segment_relationship[AS] = (
        segment_vars[AS] == 1 + segment_increase_vars[AS] - segment_decrease_vars[AS]
    )

    # 4. Asset Exposure Relationship Constraint
    asset_exposure = gp.Equation(
        m,
        domain=A,
        description="Relationship between segment ratio and asset level exposure",
    )
    asset_exposure[A] = (
        gp.Sum(AS[A, S], exposure[AS] * segment_vars[AS]) == portfolio_exposure_vars[A]
    )

    # 5. Total Exposure Relationship Constraint
    total_exposure = gp.Equation(
        m,
        description="Relationship between asset exposure and total portfolio exposure",
    )
    total_exposure[...] = gp.Sum(A, portfolio_exposure_vars[A]) == new_total_exposure

    model_constraints = m.getEquations()

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
        * exposure[AS]
        - fix_origination_cost[AS] * segment_increase_bvars[AS]
        - fix_sell_cost[AS] * segment_decrease_bvars[AS],
    )

    max_net_profit = gp.Model(
        m,
        problem=gp.Problem.MIP,
        sense=gp.Sense.MAX,
        equations=model_constraints + [def_net_profit],
        objective=net_profit,
    )

    # CVaR Model by Rockafellar and Uryasev
    VaR = gp.Variable(m, description="Value at risk")
    u = gp.Variable(m, domain=scen, type="Positive", description="Auxiliary variable for CVaR")

    def_u = gp.Equation(m, domain=scen)
    def_u[scen] = (
        u[scen]
        >= -gp.Sum(
            AS, average_profitability_scen[AS, scen] * segment_vars[AS] * exposure[AS]
        )
        - VaR
    )

    confidence_level = gp.Parameter(m, records=options.confidence_level)
    CVaR = gp.Variable(m, description="Conditional value at risk")
    def_CVaR = gp.Equation(m)
    def_CVaR[...] = CVaR == VaR + 1 / ((1 - confidence_level) * gp.Card(scen)) * gp.Sum(
        scen, u[scen]
    )

    min_CVaR = gp.Model(
        m,
        problem=gp.Problem.MIP,
        sense=gp.Sense.MIN,
        equations=model_constraints + [def_u, def_CVaR, def_net_profit],
        objective=CVaR,
    )

    return {
        "m": m,
        "A": A,
        "S": S,
        "AS": AS,
        "scen": scen,
        "exposure": exposure,
        "risk_weight": risk_weight,
        "risk_weight_limit": risk_weight_limit,
        "average_profitability": average_profitability,
        "average_profitability_scen": average_profitability_scen,
        "rel_origination_cost": rel_origination_cost,
        "rel_sell_cost": rel_sell_cost,
        "fix_origination_cost": fix_origination_cost,
        "fix_sell_cost": fix_sell_cost,
        "segment_vars": segment_vars,
        "segment_increase_vars": segment_increase_vars,
        "segment_decrease_vars": segment_decrease_vars,
        "segment_increase_bvars": segment_increase_bvars,
        "segment_decrease_bvars": segment_decrease_bvars,
        "portfolio_exposure_vars": portfolio_exposure_vars,
        "new_total_exposure": new_total_exposure,
        "net_profit": net_profit,
        "def_net_profit": def_net_profit,
        "CVaR": CVaR,
        "def_CVaR": def_CVaR,
        "def_u": def_u,
        "VaR": VaR,
        "u": u,
        "max_net_profit": max_net_profit,
        "min_CVaR": min_CVaR,
        "model_constraints": model_constraints,
    }


def _generate_results(
    model_components: dict,
    segments: pd.DataFrame,
    assets: pd.DataFrame,
    covariance_df: pd.DataFrame,
    options: nextmv.Options,
    active_model,
    maximum_net_profit: float,
    solution_pool_df: pd.DataFrame = None,
    approach_name: str = "objective_1",
) -> nextmv.Output:
    """Generate results and visualizations from a solved model."""
    m = model_components["m"]
    A = model_components["A"]
    S = model_components["S"]
    AS = model_components["AS"]
    exposure = model_components["exposure"]
    risk_weight = model_components["risk_weight"]
    risk_weight_limit = model_components["risk_weight_limit"]
    average_profitability = model_components["average_profitability"]
    rel_origination_cost = model_components["rel_origination_cost"]
    rel_sell_cost = model_components["rel_sell_cost"]
    fix_origination_cost = model_components["fix_origination_cost"]
    fix_sell_cost = model_components["fix_sell_cost"]
    segment_vars = model_components["segment_vars"]
    segment_increase_vars = model_components["segment_increase_vars"]
    segment_decrease_vars = model_components["segment_decrease_vars"]
    segment_increase_bvars = model_components["segment_increase_bvars"]
    segment_decrease_bvars = model_components["segment_decrease_bvars"]
    new_total_exposure = model_components["new_total_exposure"]
    net_profit = model_components["net_profit"]
    CVaR = model_components["CVaR"]

    pd.set_option("display.max_rows", None)
    pd.options.display.float_format = "{:,.2f}".format

    # Segment-level results
    repCol = gp.Set(
        m,
        records=[
            "old exposure",
            "new exposure",
            "rel cost incr",
            "fix cost incr",
            "rel cost decr",
            "fix cost decr",
            "profit",
            "net profit",
        ],
    )
    repAS = gp.Parameter(m, domain=[A, S, repCol])
    repAS[AS, "old exposure"] = exposure[AS]
    repAS[AS, "new exposure"] = exposure[AS] * segment_vars[AS]
    repAS[AS, "rel cost incr"] = (
        rel_origination_cost[AS] * segment_increase_vars[AS] * exposure[AS]
    )
    repAS[AS, "fix cost incr"] = fix_origination_cost[AS] * segment_increase_bvars[AS]
    repAS[AS, "rel cost decr"] = (
        rel_sell_cost[AS] * segment_decrease_vars[AS] * exposure[AS]
    )
    repAS[AS, "fix cost decr"] = fix_sell_cost[AS] * segment_decrease_bvars[AS]
    repAS[AS, "profit"] = average_profitability[AS] * segment_vars[AS] * exposure[AS]
    repAS[AS, "net profit"] = (
        repAS[AS, "profit"]
        - repAS[AS, "rel cost incr"]
        - repAS[AS, "rel cost decr"]
        - repAS[AS, "fix cost incr"]
        - repAS[AS, "fix cost decr"]
    )

    # Asset-level results
    repA = gp.Parameter(m, domain=[A, repCol])
    repA[A, repCol] = gp.Sum(AS[A, S], repAS[AS, repCol])

    # Portfolio-level summary
    rep = gp.Parameter(m, domain=["*"])
    rep[repCol] = gp.Sum(A, repA[A, repCol])
    rep["risk weight"] = gp.Sum(AS[A, S], risk_weight[AS] * exposure[AS] * segment_vars[AS])
    rep["risk weight limit"] = risk_weight_limit * new_total_exposure
    rep["CVaR"] = CVaR
    if maximum_net_profit is not None:
        rep["max possible profit"] = maximum_net_profit
    rep["confidence level"] = options.confidence_level

    # Convert to DataFrames
    segment_results = repAS.pivot()
    asset_results = repA.pivot()
    portfolio_summary = rep.records

    if options.verbose:
        print("\n=== Segment-Level Results (Detailed) ===")
        print(segment_results)
        print("\n=== Asset-Level Results ===")
        print(asset_results)

    print("\n=== Portfolio Summary ===")
    print(portfolio_summary)

    # Generate custom visualizations
    print("\n=== Generating Custom Visualizations ===")
    input_viz, output_viz = create_all_visualizations(
        segment_results=segment_results,
        asset_results=asset_results,
        portfolio_summary=portfolio_summary,
        segments_input=segments,
        assets_input=assets,
        covariance_df=covariance_df,
    )

    # Create Nextmv assets for custom visualization tabs
    visualization_assets = [
        nextmv.Asset(
            name="Input Analysis",
            content_type="json",
            visual=nextmv.Visual(
                visual_schema=nextmv.VisualSchema.PLOTLY,
                visual_type="custom-tab",
                label="Input Analysis",
            ),
            content=input_viz,
        ),
        nextmv.Asset(
            name="Rebalancing Results",
            content_type="json",
            visual=nextmv.Visual(
                visual_schema=nextmv.VisualSchema.PLOTLY,
                visual_type="custom-tab",
                label="Rebalancing Results",
            ),
            content=output_viz,
        ),
    ]
    print(f"Created {len(visualization_assets)} visualization tabs")

    # Build solution files list
    solution_files = [
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
    ]

    # Add solution pool if available
    if solution_pool_df is not None:
        solution_files.append(
            nextmv.csv_solution_file(
                name="solution_pool",
                data=solution_pool_df.to_dict(orient="records"),
            )
        )

    # Statistics
    stats = nextmv.Statistics(
        result=nextmv.ResultStatistics(
            value=net_profit.toValue(),
            duration=active_model.total_solve_time,
            custom={
                "optimized_exposure": new_total_exposure.toValue(),
                "model_status": active_model.status,
                "num_variables": active_model.num_variables,
                "num_constraints": active_model.num_equations,
                "objective_type": approach_name,
                "CVaR": CVaR.toValue(),
                "max_possible_profit": maximum_net_profit,
                "target": options.target,
                "confidence_level": options.confidence_level,
                "num_solutions": len(solution_pool_df) if solution_pool_df is not None else 1,
            },
        )
    )

    output = nextmv.Output(
        options=options,
        output_format=nextmv.OutputFormat.MULTI_FILE,
        statistics=stats,
        solution_files=solution_files,
        assets=visualization_assets,
    )

    return output


def run_stage1_profit(
    segments: pd.DataFrame,
    assets: pd.DataFrame,
    options: nextmv.Options,
) -> dict:
    """
    Stage 1: Maximize profit only.

    Returns a dict containing the maximum_net_profit value and model components
    for use in Stage 2.
    """
    print("\n=== Stage 1: Maximizing Net Profit ===")

    model_components = _create_base_model(segments, assets, options)
    max_net_profit = model_components["max_net_profit"]

    max_net_profit.solve(solver="xpress", output=sys.stdout)
    maximum_net_profit = max_net_profit.objective_value

    print(f"Maximum achievable net profit: {maximum_net_profit:.2f}")

    return {
        "maximum_net_profit": maximum_net_profit,
        "solve_time": max_net_profit.total_solve_time,
    }


def run_stage2_cvar(
    segments: pd.DataFrame,
    assets: pd.DataFrame,
    covariance_df: pd.DataFrame,
    options: nextmv.Options,
    maximum_net_profit: float,
) -> nextmv.Output:
    """
    Stage 2: Minimize CVaR with profit constraint.

    Args:
        segments: Segment data DataFrame
        assets: Asset data DataFrame
        covariance_df: Covariance matrix DataFrame
        options: Nextmv options
        maximum_net_profit: Maximum profit from Stage 1, used to set profit floor

    Returns:
        nextmv.Output with results
    """
    print(f"\n=== Stage 2: Minimizing CVaR with Solution Pool ===")
    print(f"Profit target: {options.target * 100:.1f}% of maximum = {options.target * maximum_net_profit:.2f}")

    model_components = _create_base_model(segments, assets, options)
    m = model_components["m"]
    net_profit = model_components["net_profit"]
    min_CVaR = model_components["min_CVaR"]

    # Set profit floor constraint
    net_profit.lo = options.target * maximum_net_profit

    # Suppress solution pool warnings
    logger = logging.getLogger("MODEL")
    logger.disabled = True

    # Get output directory for solution pool file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    solution_pool_gdx = os.path.join(script_dir, "outputs", "solution_pool.gdx")
    os.makedirs(os.path.dirname(solution_pool_gdx), exist_ok=True)

    min_CVaR.solve(
        solver="xpress",
        output=sys.stdout,
        solver_options={
            "solnpoolCapacity": 16,
            "solnpoolDupPolicy": 0,
            "solnpoolCullDiversity": 20,
            "solnpoolCullRounds": 10,
            "solnpoolPop": 2,
            "SolNPoolNumSym": len(m.listVariables()),
            "solnpoolMerge": solution_pool_gdx,
        },
    )
    logger.disabled = False

    # Extract solution pool data
    solution_pool_df = None
    try:
        soln = gp.Container(solution_pool_gdx)
        profit_y = soln.data["net_profit"].records["value"].to_list()
        cvar_x = soln.data["CVaR"].records["value"].to_list()
        solution_pool_df = pd.DataFrame({
            "solution_id": range(1, len(profit_y) + 1),
            "net_profit": profit_y,
            "CVaR": cvar_x,
        })
    except Exception as e:
        print(f"Warning: Could not extract solution pool: {e}")

    net_profit.lo = gp.SpecialValues.NEGINF

    return _generate_results(
        model_components=model_components,
        segments=segments,
        assets=assets,
        covariance_df=covariance_df,
        options=options,
        active_model=min_CVaR,
        maximum_net_profit=maximum_net_profit,
        solution_pool_df=solution_pool_df,
        approach_name="objective_1",
    )


def run_objective2(
    segments: pd.DataFrame,
    assets: pd.DataFrame,
    covariance_df: pd.DataFrame,
    options: nextmv.Options,
) -> nextmv.Output:
    """
    Run hierarchical multi-objective optimization (profit - 0.5*CVaR).

    Returns:
        nextmv.Output with results
    """
    print("\n=== Hierarchical Multi-Objective Optimization ===")
    print("Maximizing profit while minimizing CVaR")

    model_components = _create_base_model(segments, assets, options)
    m = model_components["m"]
    net_profit = model_components["net_profit"]
    CVaR = model_components["CVaR"]
    model_constraints = model_components["model_constraints"]
    def_u = model_components["def_u"]
    def_CVaR = model_components["def_CVaR"]
    def_net_profit = model_components["def_net_profit"]

    obj1 = gp.Variable(m, description="First hierarchical objective")
    obj2 = gp.Variable(m, description="Second hierarchical objective")
    defobj1 = gp.Equation(m, description="Definition first hierarchical objective")
    defobj2 = gp.Equation(m, description="Definition second hierarchical objective")
    defobj1[...] = obj1 == net_profit
    defobj2[...] = obj2 == CVaR

    multobj = gp.Model(
        m,
        problem=gp.Problem.MIP,
        sense=gp.Sense.MAX,
        equations=[defobj1, defobj2] + model_constraints + [def_u, def_CVaR, def_net_profit],
        objective=obj1 - 0.5 * obj2,
    )
    multobj.solve(
        solver="xpress",
        output=sys.stdout,
        solver_options={"multobj": 1, "ObjNRelTol": "obj1 0.1"},
    )

    maximum_net_profit = net_profit.toValue()

    return _generate_results(
        model_components=model_components,
        segments=segments,
        assets=assets,
        covariance_df=covariance_df,
        options=options,
        active_model=multobj,
        maximum_net_profit=maximum_net_profit,
        solution_pool_df=None,
        approach_name="objective_2",
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
