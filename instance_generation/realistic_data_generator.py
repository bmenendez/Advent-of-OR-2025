"""
Realistic Data Generator for CVaR Portfolio Optimization

This generator creates realistic input data for the third model by:
1. Fetching actual Yahoo Finance data for 19 tickers
2. Selecting 6 representative quarters (2 bull, 2 bear, 2 stable)
3. Using within-quarter differentiation for segment profitability
4. Generating fixed costs as 5-15x relative costs
"""

import pandas as pd
import numpy as np
import os
import yfinance as yf
from typing import Dict, List, Tuple


class RealisticDataGenerator:
    """Generate realistic portfolio data using Yahoo Finance historical returns."""

    # Segment quantiles for risk tier differentiation
    SEGMENT_QUANTILES = {
        'Prime': 0.40,        # Conservative - dampened returns
        'Standard': 0.50,     # Median - baseline returns
        'Substandard': 0.75,  # Moderate risk - amplified returns
        'Subprime': 0.90      # High risk - most amplified returns
    }

    # Thresholds for quarter classification
    BULL_THRESHOLD = 0.05   # > 5% mean return
    BEAR_THRESHOLD = -0.05  # < -5% mean return

    def __init__(self, seed: int = 42):
        """Initialize the generator with a random seed for reproducibility."""
        np.random.seed(seed)
        self.seed = seed

    def fetch_market_data(self, tickers: Dict[str, str],
                          start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch daily market data from Yahoo Finance.

        Args:
            tickers: Dict mapping ticker symbols to loan type names
            start_date: Start date for data fetch (YYYY-MM-DD)
            end_date: End date for data fetch (YYYY-MM-DD)

        Returns:
            DataFrame with daily closing prices and quarter labels
        """
        ticker_symbols = list(tickers.keys())
        print(f"Downloading data for {len(ticker_symbols)} tickers...")

        data = yf.download(
            ticker_symbols,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )['Close']

        data = data.reset_index()
        data['Date'] = pd.to_datetime(data['Date'])
        data['Quarter'] = data['Date'].dt.to_period('Q')

        return data

    def calculate_quarterly_returns(self, daily_data: pd.DataFrame,
                                     ticker_symbols: List[str]) -> pd.DataFrame:
        """
        Calculate quarterly returns for each ticker.

        Args:
            daily_data: DataFrame with daily prices
            ticker_symbols: List of ticker symbols

        Returns:
            DataFrame with quarterly percentage returns
        """
        # Get last price of each quarter
        quarterly_last = daily_data.groupby('Quarter')[ticker_symbols].last()

        # Calculate percentage change
        quarterly_returns = quarterly_last.pct_change().dropna()

        return quarterly_returns

    def select_representative_quarters(self, quarterly_returns: pd.DataFrame,
                                        n_quarters: int = 6) -> List:
        """
        Select representative quarters covering bull, bear, and stable conditions.

        Strategy:
        - Calculate mean return across all tickers for each quarter
        - Classify as BULL (>5%), BEAR (<-5%), or STABLE
        - Select 2 from each category

        Args:
            quarterly_returns: DataFrame with quarterly returns per ticker
            n_quarters: Number of quarters to select (default 6)

        Returns:
            List of selected quarter periods
        """
        # Calculate mean return for each quarter
        mean_returns = quarterly_returns.mean(axis=1)

        # Classify quarters
        bulls = mean_returns[mean_returns > self.BULL_THRESHOLD].sort_values(ascending=False)
        bears = mean_returns[mean_returns < self.BEAR_THRESHOLD].sort_values(ascending=True)
        stables = mean_returns[
            (mean_returns >= self.BEAR_THRESHOLD) &
            (mean_returns <= self.BULL_THRESHOLD)
        ]

        selected = []

        # Select 2 bull quarters (strongest and moderate)
        bull_list = bulls.index.tolist()
        if len(bull_list) >= 2:
            selected.append(bull_list[0])  # Strongest bull
            selected.append(bull_list[len(bull_list) // 2])  # Moderate bull
        elif len(bull_list) == 1:
            selected.append(bull_list[0])

        # Select 2 bear quarters (worst and moderate)
        bear_list = bears.index.tolist()
        if len(bear_list) >= 2:
            selected.append(bear_list[0])  # Worst bear
            selected.append(bear_list[len(bear_list) // 2])  # Moderate bear
        elif len(bear_list) == 1:
            selected.append(bear_list[0])

        # Select 2 stable quarters
        stable_list = stables.index.tolist()
        if len(stable_list) >= 2:
            # Take median and one with highest cross-asset volatility
            median_idx = len(stable_list) // 2
            selected.append(stable_list[median_idx])
            # Find stable quarter with highest volatility for diversity
            stable_vol = quarterly_returns.loc[stables.index].std(axis=1)
            high_vol_quarter = stable_vol.idxmax()
            if high_vol_quarter not in selected:
                selected.append(high_vol_quarter)
            elif len(stable_list) > 2:
                # Pick another stable quarter
                for q in stable_list:
                    if q not in selected:
                        selected.append(q)
                        break
        elif len(stable_list) == 1:
            selected.append(stable_list[0])

        # Fill remaining if we don't have enough
        all_quarters = mean_returns.sort_values().index.tolist()
        for q in all_quarters:
            if len(selected) >= n_quarters:
                break
            if q not in selected:
                selected.append(q)

        # Sort chronologically for readability
        selected = sorted(selected[:n_quarters])

        return selected

    def calculate_segment_profitability(self, daily_data: pd.DataFrame,
                                         selected_quarters: List,
                                         tickers: Dict[str, str]) -> pd.DataFrame:
        """
        Calculate segment profitability using within-quarter differentiation.

        For each quarter:
        1. Calculate the quarterly return for each ticker
        2. Apply segment differentiation factor based on quantile

        Segment factors:
        - Prime (0.40): return * 0.95 (dampened)
        - Standard (0.50): return * 1.0 (baseline)
        - Substandard (0.75): return * 1.125 (amplified)
        - Subprime (0.90): return * 1.20 (most amplified)

        Args:
            daily_data: DataFrame with daily prices
            selected_quarters: List of quarter periods to use as scenarios
            tickers: Dict mapping ticker symbols to loan names

        Returns:
            DataFrame with segment profitability for each scenario
        """
        ticker_symbols = list(tickers.keys())
        results = []

        for ticker_symbol in ticker_symbols:
            loan_type = tickers[ticker_symbol]

            for segment_name, quantile_val in self.SEGMENT_QUANTILES.items():
                row = {
                    'asset': loan_type,
                    'segment_id': f"{loan_type}_{segment_name}"
                }

                # Calculate segment factor: higher quantile = more amplified returns
                # Formula: 1 + (quantile - 0.5) * 0.5
                # Prime (0.40): 1 + (-0.10) * 0.5 = 0.95
                # Standard (0.50): 1.0
                # Substandard (0.75): 1 + (0.25) * 0.5 = 1.125
                # Subprime (0.90): 1 + (0.40) * 0.5 = 1.20
                segment_factor = 1 + (quantile_val - 0.5) * 0.5

                for i, quarter in enumerate(selected_quarters, 1):
                    # Get data for this quarter
                    quarter_mask = daily_data['Quarter'] == quarter
                    quarter_data = daily_data[quarter_mask][ticker_symbol].dropna()

                    if len(quarter_data) < 2:
                        row[f'average_profitability_{i}'] = 0.0
                        continue

                    # Calculate quarterly return
                    quarterly_return = quarter_data.iloc[-1] / quarter_data.iloc[0] - 1

                    # Apply segment differentiation
                    segment_return = quarterly_return * segment_factor

                    # Add small noise for realism (std = 1% of return magnitude)
                    noise = np.random.normal(0, max(abs(quarterly_return) * 0.01, 0.001))
                    segment_return += noise

                    row[f'average_profitability_{i}'] = round(segment_return, 4)

                results.append(row)

        return pd.DataFrame(results)

    def generate_costs(self, segments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate realistic fixed and relative costs.

        Relative costs: 0.01 - 0.07 (based on real transaction costs)
        Fixed costs: 5-15x relative costs (flat fees)

        Args:
            segments_df: DataFrame with segment data

        Returns:
            DataFrame with cost columns added
        """
        n_segments = len(segments_df)

        # Relative costs (percentage-based)
        rel_sell_cost = np.random.uniform(0.01, 0.07, n_segments)
        rel_origination_cost = np.random.uniform(0.01, 0.07, n_segments)

        # Fixed cost multiplier (5x - 15x of relative costs)
        fix_multiplier_sell = np.random.uniform(5, 15, n_segments)
        fix_multiplier_orig = np.random.uniform(5, 15, n_segments)

        fix_sell_cost = rel_sell_cost * fix_multiplier_sell
        fix_origination_cost = rel_origination_cost * fix_multiplier_orig

        segments_df['rel_sell_cost'] = np.round(rel_sell_cost, 3)
        segments_df['rel_origination_cost'] = np.round(rel_origination_cost, 3)
        segments_df['fix_sell_cost'] = np.round(fix_sell_cost, 2)
        segments_df['fix_origination_cost'] = np.round(fix_origination_cost, 2)

        return segments_df

    def generate_exposure_and_risk(self, segments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate exposure and risk_weight for each segment.

        Args:
            segments_df: DataFrame with segment data

        Returns:
            DataFrame with exposure and risk_weight columns added
        """
        n_segments = len(segments_df)

        # Exposure: random integer between 1,000 and 100,000
        segments_df['exposure'] = np.random.randint(1000, 100001, n_segments)

        # Risk weight: varies by segment type
        # Subprime segments should have higher risk weights
        risk_weights = []
        for segment_id in segments_df['segment_id']:
            if 'Prime' in segment_id and 'Subprime' not in segment_id:
                risk_weights.append(np.random.uniform(0.10, 0.25))
            elif 'Standard' in segment_id:
                risk_weights.append(np.random.uniform(0.20, 0.35))
            elif 'Substandard' in segment_id:
                risk_weights.append(np.random.uniform(0.30, 0.45))
            else:  # Subprime
                risk_weights.append(np.random.uniform(0.40, 0.50))

        segments_df['risk_weight'] = np.round(risk_weights, 2)

        return segments_df

    def generate_assets(self, tickers: Dict[str, str]) -> pd.DataFrame:
        """
        Generate assets.csv with liquidity bounds.

        Args:
            tickers: Dict mapping ticker symbols to loan names

        Returns:
            DataFrame with asset-level constraints
        """
        loan_types = list(tickers.values())
        return pd.DataFrame({
            'asset': loan_types,
            'max_exposure_decrease': np.round(np.random.uniform(0.20, 0.50, len(loan_types)), 4),
            'max_exposure_increase': np.round(np.random.uniform(0.20, 0.50, len(loan_types)), 4)
        })

    def generate_covariance(self, quarterly_returns: pd.DataFrame,
                             tickers: Dict[str, str]) -> pd.DataFrame:
        """
        Generate covariance matrix from historical returns.

        Args:
            quarterly_returns: DataFrame with quarterly returns
            tickers: Dict mapping ticker symbols to loan names

        Returns:
            DataFrame with covariance matrix
        """
        ticker_symbols = list(tickers.keys())
        loan_types = list(tickers.values())

        # Calculate covariance
        cov_matrix = quarterly_returns[ticker_symbols].cov()

        # Rename columns/index to loan types
        cov_matrix.columns = loan_types
        cov_matrix.index = loan_types

        # Convert to DataFrame format with asset column
        cov_df = cov_matrix.reset_index()
        cov_df.rename(columns={'index': 'asset'}, inplace=True)

        return cov_df

    def generate(self, tickers: Dict[str, str],
                 start_date: str = "2018-12-01",
                 end_date: str = "2025-11-30",
                 n_scenarios: int = 6,
                 output_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate complete dataset with realistic scenarios.

        Args:
            tickers: Dict mapping ticker symbols to loan type names
            start_date: Start date for Yahoo Finance data
            end_date: End date for Yahoo Finance data
            n_scenarios: Number of scenarios (quarters) to select
            output_dir: Directory to save output files (optional)

        Returns:
            Tuple of (segments_df, assets_df, covariance_df)
        """
        # Step 1: Fetch data
        print("Step 1: Fetching market data from Yahoo Finance...")
        daily_data = self.fetch_market_data(tickers, start_date, end_date)
        ticker_symbols = list(tickers.keys())

        # Step 2: Calculate quarterly returns
        print("Step 2: Calculating quarterly returns...")
        quarterly_returns = self.calculate_quarterly_returns(daily_data, ticker_symbols)
        print(f"  Found {len(quarterly_returns)} quarters of data")

        # Step 3: Select representative quarters
        print(f"Step 3: Selecting {n_scenarios} representative quarters...")
        selected_quarters = self.select_representative_quarters(quarterly_returns, n_scenarios)

        # Print selected quarters with classification
        mean_returns = quarterly_returns.mean(axis=1)
        print("\n  Selected quarters:")
        for q in selected_quarters:
            r = mean_returns[q]
            if r > self.BULL_THRESHOLD:
                classification = "BULL"
            elif r < self.BEAR_THRESHOLD:
                classification = "BEAR"
            else:
                classification = "STABLE"
            print(f"    {q}: {r:+.2%} ({classification})")
        print()

        # Step 4: Calculate segment profitability
        print("Step 4: Calculating segment profitability...")
        segments_df = self.calculate_segment_profitability(
            daily_data, selected_quarters, tickers
        )

        # Step 5: Generate costs
        print("Step 5: Generating transaction costs...")
        segments_df = self.generate_costs(segments_df)

        # Step 6: Generate exposure and risk weights
        print("Step 6: Generating exposure and risk weights...")
        segments_df = self.generate_exposure_and_risk(segments_df)

        # Reorder columns to match expected format
        profitability_cols = [f'average_profitability_{i}' for i in range(1, n_scenarios + 1)]
        column_order = [
            'asset', 'segment_id', 'exposure', 'risk_weight',
            'fix_sell_cost', 'rel_sell_cost', 'fix_origination_cost', 'rel_origination_cost'
        ] + profitability_cols
        segments_df = segments_df[column_order]

        # Step 7: Generate assets.csv
        print("Step 7: Generating assets data...")
        assets_df = self.generate_assets(tickers)

        # Step 8: Generate covariance.csv
        print("Step 8: Generating covariance matrix...")
        covariance_df = self.generate_covariance(quarterly_returns, tickers)

        # Save files if output_dir specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            segments_df.to_csv(f"{output_dir}/segments.csv", index=False)
            assets_df.to_csv(f"{output_dir}/assets.csv", index=False)
            covariance_df.to_csv(f"{output_dir}/covariance.csv", index=False)
            print(f"\nFiles saved to {output_dir}/")

        # Print summary
        print("\n" + "=" * 50)
        print("GENERATION COMPLETE")
        print("=" * 50)
        print(f"Segments: {len(segments_df)} rows, {len(segments_df.columns)} columns")
        print(f"  - Assets: {segments_df['asset'].nunique()}")
        print(f"  - Segments per asset: {len(self.SEGMENT_QUANTILES)}")
        print(f"  - Scenarios: {n_scenarios}")
        print(f"Assets: {len(assets_df)} rows")
        print(f"Covariance: {len(covariance_df)}x{len(covariance_df)} matrix")

        # Validate cost ratios
        ratios = segments_df['fix_sell_cost'] / segments_df['rel_sell_cost']
        print(f"\nCost ratio validation (fix/rel):")
        print(f"  Min: {ratios.min():.1f}x, Max: {ratios.max():.1f}x, Mean: {ratios.mean():.1f}x")

        return segments_df, assets_df, covariance_df


if __name__ == "__main__":
    # Load tickers from CSV
    tickers_path = os.path.join("instance_generation", "market_tickers.csv")
    tickers_df = pd.read_csv(tickers_path, index_col=0)
    market_tickers = tickers_df.to_dict(orient="dict")["Loan"]

    print(f"Loaded {len(market_tickers)} tickers from {tickers_path}")
    print()

    # Generate data
    generator = RealisticDataGenerator(seed=42)
    segments, assets, covariance = generator.generate(
        tickers=market_tickers,
        start_date="2018-12-01",
        end_date="2025-11-30",
        n_scenarios=6,
        output_dir=os.path.join("advent_of_or_nextmvified_third_model", "inputs")
    )
