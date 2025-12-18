"""
Custom Plotly visualizations for Portfolio Rebalancing optimization results.

This module creates visualizations that integrate with Nextmv's custom visualization
feature using Plotly charts.

Selected visualizations:
- Input: Correlation heatmap
- Output: Treemap, Butterfly, Waterfall
"""

import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def create_treemap(
    segment_results: pd.DataFrame,
    segments_input: pd.DataFrame,
) -> dict:
    """
    Create a treemap visualization showing portfolio composition with % change overlay.

    FLAGSHIP VISUALIZATION: Shows both portfolio structure AND optimization decisions
    in a single view.

    - Rectangle size: New exposure amount
    - Color: Percentage change (red = decrease, white = no change, green = increase)
    - Hierarchy: Asset -> Segment

    Args:
        segment_results: DataFrame with old/new exposure results (from optimization)
        segments_input: Input DataFrame with asset and segment_id columns

    Returns:
        Plotly figure as dictionary for Nextmv asset content
    """
    # Merge results with input to get asset/segment identifiers
    df = segments_input[["asset", "segment_id"]].copy()
    df["old_exposure"] = segment_results["old exposure"].values
    df["new_exposure"] = segment_results["new exposure"].values

    # Calculate percentage change
    df["pct_change"] = ((df["new_exposure"] - df["old_exposure"]) / df["old_exposure"] * 100)

    # For treemap display, ensure minimum value for visibility
    # Use at least 1% of total exposure for tiny segments
    min_display = df["new_exposure"].sum() * 0.001  # 0.1% of total
    df["display_value"] = df["new_exposure"].clip(lower=max(min_display, 1000))

    # Clip extreme values for better color scale (-100% to +500%)
    df["pct_change_clipped"] = df["pct_change"].clip(-100, 500)

    # Create segment labels (just the tier: Prime/Standard/etc.)
    df["segment_tier"] = df["segment_id"].str.split("_").str[-1]

    # Create hover text
    df["hover_text"] = df.apply(
        lambda row: (
            f"<b>{row['segment_id']}</b><br>"
            f"Old: ${row['old_exposure']:,.0f}<br>"
            f"New: ${row['new_exposure']:,.0f}<br>"
            f"Change: {row['pct_change']:+.1f}%"
        ),
        axis=1
    )

    # Use plotly express treemap which handles hierarchy better
    fig = px.treemap(
        df,
        path=["asset", "segment_tier"],
        values="display_value",
        color="pct_change_clipped",
        color_continuous_scale=[
            [0, "#d73027"],      # Red for -100%
            [0.167, "#fc8d59"],  # Orange
            [0.333, "#fee08b"],  # Yellow
            [0.5, "#ffffbf"],    # White/neutral for 0% (mapped to middle)
            [0.667, "#d9ef8b"],  # Light green
            [0.833, "#91cf60"],  # Green
            [1, "#1a9850"]       # Dark green for +500%
        ],
        color_continuous_midpoint=0,
        range_color=[-100, 500],
        custom_data=["segment_id", "old_exposure", "new_exposure", "pct_change"],
    )

    # Update hover template
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Old: $%{customdata[1]:,.0f}<br>"
            "New: $%{customdata[2]:,.0f}<br>"
            "Change: %{customdata[3]:+.1f}%"
            "<extra></extra>"
        ),
        textinfo="label+value",
        texttemplate="%{label}<br>$%{value:,.0f}",
    )

    fig.update_layout(
        title=dict(
            text="Portfolio Composition & Rebalancing Changes (Size = Exposure, Color = % Change)",
            font=dict(size=14),
        ),
        margin=dict(t=50, l=10, r=10, b=10),
        coloraxis_colorbar=dict(
            title="% Change",
            ticksuffix="%",
            tickvals=[-100, -50, 0, 100, 250, 500],
        ),
    )

    return json.loads(fig.to_json())


def create_butterfly(asset_results: pd.DataFrame, assets_input: pd.DataFrame) -> dict:
    """
    Create a butterfly/tornado chart showing exposure changes.

    Diverging horizontal bar chart centered at zero:
    - Assets on Y-axis
    - Bars extending left (decreases) and right (increases)
    - Shows magnitude and direction of rebalancing decisions

    Args:
        asset_results: DataFrame with old/new exposure by asset
        assets_input: Input DataFrame with asset names

    Returns:
        Plotly figure as dictionary for Nextmv asset content
    """
    assets = assets_input["asset"].tolist()
    old_exp = asset_results["old exposure"].values
    new_exp = asset_results["new exposure"].values

    # Calculate change
    change = new_exp - old_exp
    pct_change = (change / old_exp) * 100

    # Create DataFrame for sorting
    df = pd.DataFrame({
        "asset": assets,
        "change": change,
        "pct_change": pct_change,
    })

    # Sort by change magnitude
    df = df.sort_values("change", ascending=True)

    # Create colors based on positive/negative
    colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in df["change"]]

    # Create hover text
    hover_text = [
        f"Change: ${c:+,.0f}<br>({p:+.1f}%)"
        for c, p in zip(df["change"], df["pct_change"])
    ]

    fig = go.Figure(go.Bar(
        y=df["asset"],
        x=df["change"],
        orientation="h",
        marker_color=colors,
        text=[f"${c:+,.0f}" for c in df["change"]],
        textposition="outside",
        hovertext=hover_text,
        hoverinfo="text+y",
    ))

    fig.update_layout(
        title=dict(
            text="Exposure Changes by Asset (Butterfly Chart)",
            font=dict(size=16),
        ),
        xaxis=dict(
            title="Change in Exposure ($)",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="black",
        ),
        yaxis=dict(
            title="",
        ),
        showlegend=False,
        margin=dict(l=150),
    )

    return json.loads(fig.to_json())


def create_waterfall(
    asset_results: pd.DataFrame,
    assets_input: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
) -> dict:
    """
    Create a waterfall chart showing net profit contribution by asset.

    Shows how each asset contributes to the total net profit,
    sorted from highest to lowest contributor.

    Args:
        asset_results: DataFrame with asset-level results
        assets_input: Input DataFrame with asset names
        portfolio_summary: DataFrame with portfolio-level metrics

    Returns:
        Plotly figure as dictionary for Nextmv asset content
    """
    assets = assets_input["asset"].tolist()
    net_profits = asset_results["net profit"].values

    # Create DataFrame for sorting
    df = pd.DataFrame({
        "asset": assets,
        "net_profit": net_profits,
    })

    # Sort by net profit (descending - biggest contributors first)
    df = df.sort_values("net_profit", ascending=False)

    # Build waterfall data
    # All assets are "relative" contributions, with a "total" at the end
    measures = ["relative"] * len(df) + ["total"]
    x_labels = df["asset"].tolist() + ["Total Net Profit"]
    y_values = df["net_profit"].tolist() + [0]  # Total is computed automatically

    # Create text labels
    text_labels = [f"${v:+,.0f}" for v in df["net_profit"]] + [
        f"${df['net_profit'].sum():,.0f}"
    ]

    # Create colors - green for positive, red for negative
    colors = []
    for v in df["net_profit"]:
        if v >= 0:
            colors.append("#2ecc71")  # Green
        else:
            colors.append("#e74c3c")  # Red

    fig = go.Figure(go.Waterfall(
        name="Net Profit by Asset",
        orientation="v",
        measure=measures,
        x=x_labels,
        y=y_values,
        text=text_labels,
        textposition="outside",
        textfont=dict(size=9),
        connector=dict(line=dict(color="rgb(63, 63, 63)", width=1)),
        increasing=dict(marker=dict(color="#2ecc71")),
        decreasing=dict(marker=dict(color="#e74c3c")),
        totals=dict(marker=dict(color="#3498db")),
    ))

    fig.update_layout(
        title=dict(
            text="Net Profit Contribution by Asset (Waterfall)",
            font=dict(size=16),
        ),
        xaxis=dict(
            title="",
            tickangle=45,
        ),
        yaxis=dict(
            title="Net Profit ($)",
            tickformat="$,.0f",
        ),
        showlegend=False,
        margin=dict(b=120),  # More space for rotated labels
    )

    return json.loads(fig.to_json())


def create_correlation_heatmap(covariance_df: pd.DataFrame) -> dict:
    """
    Create a heatmap visualization of the asset covariance/correlation matrix.

    - Color scale showing correlation strength
    - Asset labels on both axes
    - Shows diversification relationships in input data

    Args:
        covariance_df: DataFrame with covariance matrix (index and columns are asset names)

    Returns:
        Plotly figure as dictionary for Nextmv asset content
    """
    # Convert covariance to correlation for better interpretability
    # Correlation = Cov(X,Y) / (std(X) * std(Y))
    # For the diagonal, this gives 1
    std_devs = np.sqrt(np.diag(covariance_df.values))
    correlation = covariance_df.values / np.outer(std_devs, std_devs)

    # Handle any NaN/Inf values
    correlation = np.nan_to_num(correlation, nan=0, posinf=1, neginf=-1)

    assets = covariance_df.columns.tolist()

    # Create hover text
    hover_text = []
    for i, row_asset in enumerate(assets):
        row_text = []
        for j, col_asset in enumerate(assets):
            row_text.append(
                f"{row_asset} vs {col_asset}<br>"
                f"Correlation: {correlation[i, j]:.4f}<br>"
                f"Covariance: {covariance_df.values[i, j]:.6f}"
            )
        hover_text.append(row_text)

    fig = go.Figure(go.Heatmap(
        z=correlation,
        x=assets,
        y=assets,
        colorscale="RdBu_r",
        zmid=0,
        text=hover_text,
        hoverinfo="text",
        colorbar=dict(
            title="Correlation",
            tickformat=".2f",
        ),
    ))

    fig.update_layout(
        title=dict(
            text="Asset Correlation Matrix (Input Data)",
            font=dict(size=16),
        ),
        xaxis=dict(
            title="",
            tickangle=45,
        ),
        yaxis=dict(
            title="",
            autorange="reversed",
        ),
        width=700,
        height=700,
    )

    return json.loads(fig.to_json())


def create_all_visualizations(
    segment_results: pd.DataFrame,
    asset_results: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    segments_input: pd.DataFrame,
    assets_input: pd.DataFrame,
    covariance_df: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    """
    Create all visualizations and return them organized by tab.

    Args:
        segment_results: Segment-level optimization results
        asset_results: Asset-level optimization results
        portfolio_summary: Portfolio-level summary metrics
        segments_input: Input segments DataFrame
        assets_input: Input assets DataFrame
        covariance_df: Input covariance matrix

    Returns:
        Tuple of (input_visualizations, output_visualizations)
        Each is a list of Plotly figure dictionaries
    """
    # Input analysis visualizations
    input_viz = [
        create_correlation_heatmap(covariance_df),
    ]

    # Rebalancing results visualizations (treemap first as flagship)
    output_viz = [
        create_treemap(segment_results, segments_input),
        create_butterfly(asset_results, assets_input),
        create_waterfall(asset_results, assets_input, portfolio_summary),
    ]

    return input_viz, output_viz
