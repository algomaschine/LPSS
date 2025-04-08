import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import glob
from multiprocessing import Pool, cpu_count
import time
import traceback
from collections import namedtuple
import argparse

def hp_filter(y, lambda_=1600):
    """
    Hodrick-Prescott filter implementation.
    y: time series
    lambda_: smoothing parameter (1600 is standard for quarterly data, 
             for daily data we might want to adjust this)
    """
    n = len(y)
    D = np.zeros((n-2, n))
    for i in range(n-2):
        D[i, i] = 1
        D[i, i+1] = -2
        D[i, i+2] = 1
    
    trend = np.linalg.solve(np.eye(n) + lambda_ * D.T @ D, y)
    cycle = y - trend
    return trend, cycle

# --- Configuration ---
DATA_DIR = "data"  # Directory containing 1-minute CSV files
REPORTS_DIR = "reports_LPPLS_inverse" # Directory to save HTML reports
# Files should be named 'ASSET_1m_source.csv'
INDEX_HTML_FILE = 'index_LPPLS_inverse.html' # Main index file

# --- LPPLS Fitting Configuration ---
# Adjust these parameters carefully!
LPPLS_WINDOW_SIZE = 90 # Days: Length of the window for each fit (e.g., 60-120 common)
LPPLS_STEP_SIZE = 5    # Days: How often to perform a fit (1 = daily, 5 = every 5 days)
# Bounds for LPPLS parameters [tc, m, omega, A, B, C]
# tc is relative to the *start* of the window for fitting convenience
LPPLS_BOUNDS = [
    (LPPLS_WINDOW_SIZE * 0.8, LPPLS_WINDOW_SIZE * 2.0), # tc (relative end time): Must be after window end, but not too far
    (0.01, 0.99),  # m: Power law exponent (0 < m < 1 typical)
    (2.0, 25.0),   # omega: Log-frequency (2 < omega < 25 typical)
    (None, None), # A: Constant term in log-price (unbounded)
    (-1.0, -1e-6), # B: Amplitude of power law (must be < 0 for bubble)
    (None, None)  # C: Amplitude of oscillations (unbounded)
]
# Initial guess factor for tc (relative to window end)
TC_INITIAL_GUESS_FACTOR = 1.1

# --- Confidence Filtering ---
# Basic filters for qualifying a fit as potentially significant
CONFIDENCE_FILTERS = {
    'max_tc_lookahead': 60, # Days: Max estimated days until tc from window end
    'm_range': (0.1, 0.9),
    'omega_range': (4, 15) # Tighter range often used
}

NUM_PROCESSES = max(1, cpu_count() - 3) # Adjusted CPU count

# Plotting Styles (Simplified - only need one marker type now)
lppls_fit_marker = {
    'symbol': 'diamond',
    'size': 8,
    'color': 'red', # Color for confident fit markers
    'text': 'Confident LPPLS Fit Detected'
}

# --- Command Line Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(description='LPPLS Analysis Tool')
    parser.add_argument('--daily-file', type=str, help='Path to a daily CSV file for analysis. If provided, only this file will be analyzed.')
    return parser.parse_args()

# --- Daily Data Validation ---
def validate_daily_data(df, filepath):
    """Validates that the data is in daily format with correct column names and time intervals."""
    required_columns = ['Date', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Input file '{filepath}' is missing required columns: {missing_columns}")
        print("Required columns are: Date, close, volume")
        return False
    
    # Check if Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        print("Error: 'Date' column must be in datetime format")
        return False
    
    # Sort by date to ensure proper time difference calculation
    df = df.sort_values('Date')
    
    # Calculate time differences between consecutive rows
    time_diffs = df['Date'].diff().dt.total_seconds()
    
    # Check if all differences are approximately 24 hours (86400 seconds)
    # Allow for small variations (e.g., due to daylight savings)
    valid_diffs = time_diffs.between(86300, 86500)  # Allow ±5 minutes variation
    if not valid_diffs.all():
        invalid_diffs = time_diffs[~valid_diffs]
        print(f"Error: Time differences between rows should be approximately 24 hours")
        print(f"Found invalid time differences at positions: {invalid_diffs.index.tolist()}")
        print(f"Time differences: {invalid_diffs.tolist()}")
        return False
    
    return True

# --- Data Loading Function for Daily Files ---
def load_daily_file(filepath):
    """Loads and validates a daily CSV file."""
    try:
        print(f"Reading daily data from: {filepath}")
        df = pd.read_csv(filepath)
        
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Validate the data
        if not validate_daily_data(df, filepath):
            return None, None
        
        # Use the full filename (without extension) as the asset name
        asset_name = os.path.splitext(os.path.basename(filepath))[0]
        
        # Calculate inverse price and its variations
        df['raw_inverse'] = 1 / df['close']
        price_range = df['close'].max() - df['close'].min()
        inverse_range = df['raw_inverse'].max() - df['raw_inverse'].min()
        df['inverse_close'] = (df['raw_inverse'] - df['raw_inverse'].min()) * (price_range / inverse_range) + df['close'].min()
        
        # Calculate rolling mean detrended inverse price
        trend_window = min(60, len(df) // 4)
        df['inverse_trend'] = df['inverse_close'].rolling(window=trend_window, min_periods=1).mean()
        df['inverse_detrended'] = df['inverse_close'] - df['inverse_trend']
        detrended_range = df['inverse_detrended'].max() - df['inverse_detrended'].min()
        df['inverse_detrended'] = (df['inverse_detrended'] - df['inverse_detrended'].min()) * (price_range / detrended_range) + df['close'].min()
        
        # Calculate HP-filtered inverse price
        lambda_daily = 129600
        inverse_trend_hp, inverse_cycle_hp = hp_filter(df['inverse_close'].values, lambda_=lambda_daily)
        df['inverse_trend_hp'] = inverse_trend_hp
        df['inverse_cycle_hp'] = inverse_cycle_hp
        cycle_range = df['inverse_cycle_hp'].max() - df['inverse_cycle_hp'].min()
        df['inverse_cycle_hp'] = (df['inverse_cycle_hp'] - df['inverse_cycle_hp'].min()) * (price_range / cycle_range) + df['close'].min()
        
        return df, asset_name
        
    except Exception as e:
        print(f"Error loading daily file {filepath}: {e}")
        return None, None

# --- Data Loading and Resampling Function (Modified for lowercase volume) ---
def load_and_resample_daily(filepath):
    """Loads 1-min CSV, checks for 'date', 'close', 'volume', resamples to daily."""
    asset_name = "Unknown"
    try:
        basename = os.path.basename(filepath)
        asset_name = basename.split('_')[0]
        print(f"[{asset_name}] Reading data from: {filepath}")

        # Check if daily cache file exists and is recent
        daily_cache_dir = os.path.join(DATA_DIR, "daily_cache")
        os.makedirs(daily_cache_dir, exist_ok=True)
        daily_cache_file = os.path.join(daily_cache_dir, f"{asset_name}_daily.csv")
        
        # Check if cache file exists and is recent (within 1 hour)
        if os.path.exists(daily_cache_file):
            cache_mtime = os.path.getmtime(daily_cache_file)
            current_time = time.time()
            if current_time - cache_mtime <= 3600:  # 1 hour in seconds
                print(f"[{asset_name}] Using cached daily data from {daily_cache_file}")
                df_daily = pd.read_csv(daily_cache_file)
                df_daily['Date'] = pd.to_datetime(df_daily['Date'])
                return df_daily, asset_name

        # If no recent cache, proceed with 1-minute data processing
        print(f"[{asset_name}] No recent cache found, processing 1-minute data...")

        # Read header to check columns case-insensitively
        try:
            df_cols = pd.read_csv(filepath, nrows=0).columns.str.lower().tolist()
            col_mapping = {col.lower(): col for col in pd.read_csv(filepath, nrows=0).columns}
        except Exception as e:
            print(f"Warning [{asset_name}]: Could not read header from {filepath}: {e}. Skipping.")
            return None, asset_name

        required_cols_lower = ['date', 'close', 'volume']
        if not all(col in df_cols for col in required_cols_lower):
            missing = [col for col in required_cols_lower if col not in df_cols]
            print(f"Warning [{asset_name}]: Input CSV missing required columns (case-insensitive): {missing}. Required: {required_cols_lower}. Found: {df_cols}. Skipping.")
            return None, asset_name

        # Use the detected original column names
        date_col = col_mapping['date']
        close_col = col_mapping['close']
        volume_col = col_mapping['volume']

        df_1m = pd.read_csv(filepath, usecols=[date_col, close_col, volume_col])
        # Rename columns to standard lowercase for consistency
        df_1m.rename(columns={date_col: 'Date', close_col: 'close', volume_col: 'volume'}, inplace=True)

        print(f"[{asset_name}] Initial rows: {len(df_1m)}")

        df_1m['Date'] = pd.to_datetime(df_1m['Date'])
        df_1m = df_1m.dropna(subset=['Date', 'close', 'volume']) # Drop NaNs in essential columns
        df_1m = df_1m[df_1m['close'] > 1e-9] # Filter out zero/negative prices
        df_1m = df_1m.sort_values('Date').set_index('Date')

        print(f"[{asset_name}] Resampling to daily...")
        daily_agg = df_1m.resample('D', label='right', closed='right').agg({
            'close': 'last',
            'volume': 'sum' # Aggregate volume by summing
        })
        # Drop days with NaN close price *after* resampling
        df_daily = daily_agg.dropna(subset=['close'])

        # Calculate normalized inverse price
        # First calculate raw inverse
        df_daily['raw_inverse'] = 1 / df_daily['close']
        # Normalize to have similar scale as original price
        price_range = df_daily['close'].max() - df_daily['close'].min()
        inverse_range = df_daily['raw_inverse'].max() - df_daily['raw_inverse'].min()
        df_daily['inverse_close'] = (df_daily['raw_inverse'] - df_daily['raw_inverse'].min()) * (price_range / inverse_range) + df_daily['close'].min()

        # Calculate rolling mean detrended inverse price
        trend_window = min(60, len(df_daily) // 4)  # Use 60 days or 1/4 of data length, whichever is smaller
        df_daily['inverse_trend'] = df_daily['inverse_close'].rolling(window=trend_window, min_periods=1).mean()
        df_daily['inverse_detrended'] = df_daily['inverse_close'] - df_daily['inverse_trend']
        # Normalize the detrended series
        detrended_range = df_daily['inverse_detrended'].max() - df_daily['inverse_detrended'].min()
        df_daily['inverse_detrended'] = (df_daily['inverse_detrended'] - df_daily['inverse_detrended'].min()) * (price_range / detrended_range) + df_daily['close'].min()

        # Calculate HP-filtered inverse price
        # For daily data, we use a higher lambda value (129600 = 1600 * 9^4, as suggested by Ravn and Uhlig, 2002)
        lambda_daily = 129600
        inverse_trend_hp, inverse_cycle_hp = hp_filter(df_daily['inverse_close'].values, lambda_=lambda_daily)
        df_daily['inverse_trend_hp'] = inverse_trend_hp
        df_daily['inverse_cycle_hp'] = inverse_cycle_hp
        # Normalize the HP-filtered cycle
        cycle_range = df_daily['inverse_cycle_hp'].max() - df_daily['inverse_cycle_hp'].min()
        df_daily['inverse_cycle_hp'] = (df_daily['inverse_cycle_hp'] - df_daily['inverse_cycle_hp'].min()) * (price_range / cycle_range) + df_daily['close'].min()

        # Reset index to make Date a column
        df_daily = df_daily.reset_index()
        
        if df_daily.empty:
            print(f"Warning [{asset_name}]: No daily data left after resampling/cleaning.")
            return None, asset_name

        print(f"[{asset_name}] Resampled to {len(df_daily)} daily data points.")

        # Save to cache file
        print(f"[{asset_name}] Saving daily data to cache: {daily_cache_file}")
        df_daily.to_csv(daily_cache_file, index=False)
        
        return df_daily, asset_name

    except FileNotFoundError:
        print(f"Error [{asset_name}]: File not found at '{filepath}'. Skipping.")
        return None, asset_name
    except ValueError as ve:
        print(f"Error [{asset_name}]: Data validation error - {ve}. Skipping.")
        return None, asset_name
    except Exception as e:
        print(f"Error [{asset_name}]: Unexpected error during data loading/resampling for {filepath}: {e}")
        return None, asset_name

# --- LPPLS Model and Fitting Functions ---

def lppls_model_equation(t, tc, m, omega, A, B, C):
    """Calculates log-price based on LPPLS parameters."""
    dt = tc - t
    # Ensure dt is positive; return NaN or large number if t >= tc
    # Using np.maximum avoids log(<=0) errors
    log_dt = np.log(np.maximum(dt, 1e-9)) # Avoid log(0) or log(-)
    power_term = np.power(np.maximum(dt, 1e-9), m)

    # Calculate model value, handle cases where t might be >= tc by checking dt
    model_val = np.where(
        dt > 1e-9,
        A + B * power_term + C * power_term * np.cos(omega * log_dt), # phi is absorbed into C/omega shift
        A # Or np.nan or a large penalty if t >= tc within the fitting window
    )
    return model_val

def lppls_objective_function(params, t_window, log_price_window):
    """Objective function: sum of squared errors for LPPLS fit."""
    tc_rel, m, omega, A, B, C = params
    # tc_rel is relative to the START of the window, convert t_window accordingly
    t_adjusted = t_window - t_window[0] # Time starting from 0 within the window
    tc_adjusted = tc_rel # tc is now also relative to window start

    model_log_price = lppls_model_equation(t_adjusted, tc_adjusted, m, omega, A, B, C)

    # Handle potential NaNs or Infs in model output
    if np.any(np.isnan(model_log_price)) or np.any(np.isinf(model_log_price)):
        return 1e12 # Return large error if model calculation fails

    # Calculate Sum of Squared Errors
    sse = np.sum((model_log_price - log_price_window)**2)

    return sse

# Named tuple for storing results
LPPLSFitResult = namedtuple("LPPLSFitResult", ["window_end_date", "success", "tc", "m", "omega", "A", "B", "C", "sse", "message", "confident"])

def fit_lppls_window(t_window, log_price_window, window_end_date):
    """
    Fits the LPPLS model to a single window of data.
    t_window: time indices (e.g., integer days from start)
    log_price_window: log of closing prices for the window
    window_end_date: The actual date corresponding to the end of this window
    """
    if len(t_window) != len(log_price_window):
        raise ValueError("Time and log-price windows must have the same length.")
    if len(t_window) < 10: # Need sufficient points for fitting
        print("Warning: Window too short for fitting.")
        return LPPLSFitResult(window_end_date, False, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.inf, "Window too short", False)

    # --- Initial Guesses ---
    # tc_guess relative to window start
    tc_initial_rel = len(t_window) * TC_INITIAL_GUESS_FACTOR
    m_initial = 0.5
    omega_initial = 7.0 # Common starting point
    # Fit line to log_price to estimate A and maybe B's scale roughly
    try:
         poly_coeffs = np.polyfit(t_window, log_price_window, 1)
         A_initial = poly_coeffs[1] # Intercept at t=0 of window
         # B_initial requires more care, related to curvature. Use default negative.
    except: # Handle polyfit failure on constant data etc.
        A_initial = np.mean(log_price_window)

    B_initial = -0.1 # Must be negative for bubble
    C_initial = 0.0 # Start with no oscillation amplitude

    initial_guess = [tc_initial_rel, m_initial, omega_initial, A_initial, B_initial, C_initial]

    # --- Bounds ---
    # Adjust tc bound to be relative to window start
    current_bounds = [
        (len(t_window), LPPLS_BOUNDS[0][1]), # tc_rel: must be >= window end, use second part of config bound
        LPPLS_BOUNDS[1], # m
        LPPLS_BOUNDS[2], # omega
        LPPLS_BOUNDS[3], # A
        LPPLS_BOUNDS[4], # B
        LPPLS_BOUNDS[5]  # C
    ]

    # --- Optimization ---
    try:
        result = minimize(
            lppls_objective_function,
            initial_guess,
            args=(t_window, log_price_window),
            method='SLSQP', # Supports bounds and constraints
            bounds=current_bounds,
            options={'maxiter': 500, 'ftol': 1e-7, 'disp': False} # Adjust options as needed
        )

        if result.success:
            # Extract fitted parameters
            tc_fit_rel, m_fit, omega_fit, A_fit, B_fit, C_fit = result.x
            sse_fit = result.fun

            # Convert relative tc back to absolute date estimate
            tc_fit_abs_index = t_window[0] + tc_fit_rel # Index position of tc relative to original data start

            # Basic qualification check
            confident_flag = qualify_lppls_fit(result.x, len(t_window))

            return LPPLSFitResult(window_end_date, True, tc_fit_abs_index, m_fit, omega_fit, A_fit, B_fit, C_fit, sse_fit, result.message, confident_flag)
        else:
            return LPPLSFitResult(window_end_date, False, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.inf, result.message, False)

    except Exception as e:
        print(f"Error during LPPLS fit for window ending {window_end_date}: {e}")
        return LPPLSFitResult(window_end_date, False, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.inf, str(e), False)

def qualify_lppls_fit(params, window_length_points):
    """ Basic check if fitted parameters seem reasonable for a 'confident' bubble signal."""
    tc_rel, m, omega, A, B, C = params
    tc_days_from_win_end = tc_rel - window_length_points

    # Check parameter ranges and tc proximity
    m_ok = CONFIDENCE_FILTERS['m_range'][0] <= m <= CONFIDENCE_FILTERS['m_range'][1]
    omega_ok = CONFIDENCE_FILTERS['omega_range'][0] <= omega <= CONFIDENCE_FILTERS['omega_range'][1]
    tc_ok = 0 < tc_days_from_win_end <= CONFIDENCE_FILTERS['max_tc_lookahead']
    # B must be negative (enforced by bounds but double check)
    b_ok = B < 0

    return m_ok and omega_ok and tc_ok and b_ok

# --- Plotting Function (Modified for LPPLS with inverse price) ---
def generate_plot(daily_data, lppls_results_regular, lppls_results_inverse, lppls_results_detrended, lppls_results_inverse_hp, output_filepath, asset_name):
    """Generates a single HTML file with four separate 4-row plots."""
    try:
        if daily_data.empty:
            print(f"Warning [{asset_name}]: No daily data available for plotting.")
            return False

        print(f"[{asset_name}] Generating combined LPPLS plot...")

        # Create a figure with 4 separate subplot groups
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=False, vertical_spacing=0.1,
            subplot_titles=(
                f"{asset_name} - Regular Price Analysis",
                f"{asset_name} - Inverse Price Analysis",
                f"{asset_name} - Rolling Mean Detrended Analysis",
                f"{asset_name} - HP-Filtered Analysis"
            ),
            row_heights=[0.25, 0.25, 0.25, 0.25]
        )

        # Helper function to add traces for each analysis type
        def add_analysis_traces(df, results, row, price_col, title_prefix):
            # Create a subplot for this analysis type
            subplot = make_subplots(
                rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                subplot_titles=(
                    f"{title_prefix} Price & Confident LPPLS Fits",
                    "Estimated Days to Critical Time (tc - t_end)",
                    "Exponent (m)",
                    "Log-Frequency (omega)"
                ),
                row_heights=[0.4, 0.2, 0.2, 0.2]
            )

            # Panel 1: Price and Confident Fits
            subplot.add_trace(go.Scatter(
                x=df['Date'], y=df[price_col], mode='lines',
                name=f"{title_prefix} Price", line={'color': 'blue', 'width': 1.5}
            ), row=1, col=1)

            if results:
                results_df = pd.DataFrame(results)
                results_df['Date'] = pd.to_datetime(results_df['window_end_date'])
                
                # Calculate days remaining to critical time
                date_to_index = pd.Series(index=df['Date'], data=np.arange(len(df)))
                window_end_indices = results_df['Date'].map(date_to_index)
                results_df['tc_days_remaining'] = results_df['tc'] - window_end_indices
                
                confident_fits = results_df[results_df['confident'] == True]
                
                if not confident_fits.empty:
                    confident_fit_prices = df.set_index('Date').loc[confident_fits['Date']][price_col]
                    hover_texts = [
                        f"<b>Confident LPPLS Fit</b><br>Date: {date.strftime('%Y-%m-%d')}<br>"
                        f"Est. Days to tc: {tc_rem:.1f}<br>m: {m:.3f}<br>ω: {om:.2f}<br>Price: {price:.4f}"
                        for date, tc_rem, m, om, price in zip(
                            confident_fits['Date'], confident_fits['tc_days_remaining'],
                            confident_fits['m'], confident_fits['omega'], confident_fit_prices
                        )
                    ]
                    subplot.add_trace(go.Scatter(
                        x=confident_fits['Date'], y=confident_fit_prices,
                        mode='markers', name="Confident Fit",
                        marker=dict(
                            color=lppls_fit_marker['color'],
                            symbol=lppls_fit_marker['symbol'],
                            size=lppls_fit_marker['size'],
                            line=dict(width=1, color='DarkSlateGrey')
                        ),
                        text=hover_texts, hoverinfo='text'
                    ), row=1, col=1)

                # Panel 2: Days to tc
                subplot.add_trace(go.Scatter(
                    x=results_df['Date'], y=results_df['tc_days_remaining'],
                    mode='lines+markers', name="Days to tc",
                    line={'color': 'green', 'width': 1}, marker={'size': 3}
                ), row=2, col=1)
                subplot.add_hline(y=0, line_dash='dash', line_color='grey', line_width=1, row=2, col=1)

                # Panel 3: Exponent m
                subplot.add_trace(go.Scatter(
                    x=results_df['Date'], y=results_df['m'],
                    mode='lines+markers', name="Exponent m",
                    line={'color': 'purple', 'width': 1}, marker={'size': 3}
                ), row=3, col=1)
                subplot.add_hline(y=CONFIDENCE_FILTERS['m_range'][0], line_dash='dot', line_color='grey', line_width=1, row=3, col=1)
                subplot.add_hline(y=CONFIDENCE_FILTERS['m_range'][1], line_dash='dot', line_color='grey', line_width=1, row=3, col=1)

                # Panel 4: Log-Frequency omega
                subplot.add_trace(go.Scatter(
                    x=results_df['Date'], y=results_df['omega'],
                    mode='lines+markers', name="Log-Frequency ω",
                    line={'color': 'orange', 'width': 1}, marker={'size': 3}
                ), row=4, col=1)
                subplot.add_hline(y=CONFIDENCE_FILTERS['omega_range'][0], line_dash='dot', line_color='grey', line_width=1, row=4, col=1)
                subplot.add_hline(y=CONFIDENCE_FILTERS['omega_range'][1], line_dash='dot', line_color='grey', line_width=1, row=4, col=1)

            # Update subplot layout
            subplot.update_layout(
                height=950,
                showlegend=True,
                legend_title_text='Metrics & Fits',
                legend=dict(traceorder='reversed'),
                margin=dict(t=50, b=50)
            )

            # Update y-axis titles
            subplot.update_yaxes(title_text="Price / Fits", row=1, col=1)
            subplot.update_yaxes(title_text="Days to tc", range=[-10, CONFIDENCE_FILTERS['max_tc_lookahead'] * 1.5], row=2, col=1)
            subplot.update_yaxes(title_text="m", range=[0, 1], row=3, col=1)
            subplot.update_yaxes(title_text="ω", range=[0, LPPLS_BOUNDS[2][1]*1.1], row=4, col=1)
            subplot.update_xaxes(title_text="Date", row=4, col=1)

            # Add spikes and rangeslider
            subplot.update_xaxes(showspikes=True, spikecolor="grey", spikesnap="cursor", spikemode="across", spikedash='dot', spikethickness=1)
            subplot.update_yaxes(showspikes=True, spikecolor="grey", spikesnap="cursor", spikethickness=0.5)
            subplot.update_xaxes(rangeslider_visible=True, row=4, col=1)

            return subplot

        # Create subplots for each analysis type
        regular_plot = add_analysis_traces(daily_data, lppls_results_regular, 1, 'close', "Regular")
        inverse_plot = add_analysis_traces(daily_data, lppls_results_inverse, 2, 'inverse_close', "Inverse")
        detrended_plot = add_analysis_traces(daily_data, lppls_results_detrended, 3, 'inverse_detrended', "Detrended")
        hp_plot = add_analysis_traces(daily_data, lppls_results_inverse_hp, 4, 'inverse_cycle_hp', "HP-Filtered")

        # Combine all plots into a single HTML file
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{asset_name} - Combined LPPLS Analysis</title>
                <meta charset="UTF-8">
                <style>
                    .plot-container {{
                        margin-bottom: 50px;
                    }}
                </style>
            </head>
            <body>
                <h1>{asset_name} - Combined LPPLS Analysis</h1>
                <div class="plot-container">
                    {regular_plot}
                </div>
                <div class="plot-container">
                    {inverse_plot}
                </div>
                <div class="plot-container">
                    {detrended_plot}
                </div>
                <div class="plot-container">
                    {hp_plot}
                </div>
            </body>
            </html>
            """.format(
                asset_name=asset_name,
                regular_plot=regular_plot.to_html(full_html=False),
                inverse_plot=inverse_plot.to_html(full_html=False),
                detrended_plot=detrended_plot.to_html(full_html=False),
                hp_plot=hp_plot.to_html(full_html=False)
            ))

        print(f"[{asset_name}] Successfully generated combined plot: {output_filepath}")
        return True

    except Exception as e:
        print(f"Error [{asset_name}]: An error occurred during plot generation: {e}")
        return False

# --- Analysis Function (Modified for LPPLS with inverse price) ---
def analyze_asset(asset_name, daily_data, output_dir):
    """Runs the rolling LPPLS analysis pipeline for a single asset."""
    print(f"\n--- Analyzing Asset: {asset_name} (LPPLS) ---")
    output_filepath = os.path.join(output_dir, f"{asset_name}_lppls_combined.html")

    if daily_data is None or daily_data.empty or len(daily_data) < LPPLS_WINDOW_SIZE:
        print(f"Warning [{asset_name}]: Insufficient data ({len(daily_data) if daily_data is not None else 0} points) for LPPLS analysis (Window={LPPLS_WINDOW_SIZE}). Skipping.")
        return asset_name, False

    # Prepare data for fitting
    daily_data = daily_data.sort_values('Date').reset_index(drop=True)
    time_indices = np.arange(len(daily_data))

    # Fit LPPLS to regular price
    log_prices = np.log(daily_data['close'].values)
    lppls_results_regular = []
    for i in range(LPPLS_WINDOW_SIZE - 1, len(daily_data), LPPLS_STEP_SIZE):
        window_start_index = i - (LPPLS_WINDOW_SIZE - 1)
        window_end_index = i + 1
        t_win = time_indices[window_start_index:window_end_index]
        log_p_win = log_prices[window_start_index:window_end_index]
        window_end_date = daily_data['Date'].iloc[i]
        if np.std(log_p_win) >= 1e-6:
            fit_result = fit_lppls_window(t_win, log_p_win, window_end_date)
            lppls_results_regular.append(fit_result)

    # Fit LPPLS to inverse price
    log_inverse = np.log(daily_data['inverse_close'].values)
    lppls_results_inverse = []
    for i in range(LPPLS_WINDOW_SIZE - 1, len(daily_data), LPPLS_STEP_SIZE):
        window_start_index = i - (LPPLS_WINDOW_SIZE - 1)
        window_end_index = i + 1
        t_win = time_indices[window_start_index:window_end_index]
        log_p_win = log_inverse[window_start_index:window_end_index]
        window_end_date = daily_data['Date'].iloc[i]
        if np.std(log_p_win) >= 1e-6:
            fit_result = fit_lppls_window(t_win, log_p_win, window_end_date)
            lppls_results_inverse.append(fit_result)

    # Fit LPPLS to detrended inverse price
    log_detrended = np.log(daily_data['inverse_detrended'].values)
    lppls_results_detrended = []
    for i in range(LPPLS_WINDOW_SIZE - 1, len(daily_data), LPPLS_STEP_SIZE):
        window_start_index = i - (LPPLS_WINDOW_SIZE - 1)
        window_end_index = i + 1
        t_win = time_indices[window_start_index:window_end_index]
        log_p_win = log_detrended[window_start_index:window_end_index]
        window_end_date = daily_data['Date'].iloc[i]
        if np.std(log_p_win) >= 1e-6:
            fit_result = fit_lppls_window(t_win, log_p_win, window_end_date)
            lppls_results_detrended.append(fit_result)

    # Fit LPPLS to HP-filtered inverse price
    log_hp = np.log(daily_data['inverse_cycle_hp'].values)
    lppls_results_inverse_hp = []
    for i in range(LPPLS_WINDOW_SIZE - 1, len(daily_data), LPPLS_STEP_SIZE):
        window_start_index = i - (LPPLS_WINDOW_SIZE - 1)
        window_end_index = i + 1
        t_win = time_indices[window_start_index:window_end_index]
        log_p_win = log_hp[window_start_index:window_end_index]
        window_end_date = daily_data['Date'].iloc[i]
        if np.std(log_p_win) >= 1e-6:
            fit_result = fit_lppls_window(t_win, log_p_win, window_end_date)
            lppls_results_inverse_hp.append(fit_result)

    # Generate combined plot
    plot_success = generate_plot(
        daily_data,
        lppls_results_regular,
        lppls_results_inverse,
        lppls_results_detrended,
        lppls_results_inverse_hp,
        output_filepath,
        asset_name
    )

    return asset_name, plot_success

# --- Index HTML Generation (Modified for LPPLS with inverse price) ---
def create_index_html(report_files, output_filename):
    """Creates the index.html file with links and explanations for LPPLS."""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LPPLS Bubble Analysis Reports (Regular & Inverse Price)</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; line-height: 1.6; }}
        h1, h2 {{ color: #333; }}
        ul {{ list-style-type: none; padding: 0; }}
        li {{ margin-bottom: 10px; background-color: #f4f4f4; padding: 10px; border-radius: 5px; }}
        a {{ text-decoration: none; color: #007bff; font-weight: bold; }}
        a:hover {{ text-decoration: underline; }}
        .explanation {{ margin-top: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; }}
        code {{ background-color: #eee; padding: 2px 4px; border-radius: 3px; }}
    </style>
</head>
<body>
    <h1>Financial Bubble Analysis Reports (LPPLS Method - Regular & Inverse Price)</h1>
    <p>This page provides links to individual asset reports generated using the Log-Periodic Power Law Singularity (LPPLS) model, analyzing both regular and inverse price series.</p>

    <h2>Available Reports:</h2>
    <ul>
"""
    if report_files:
        for report_name, report_path in sorted(report_files.items()):
            relative_path = os.path.join(REPORTS_DIR, os.path.basename(report_path))
            html_content += f'        <li><a href="{relative_path}">{report_name}</a></li>\n'
    else:
        html_content += "        <li>No reports were generated successfully.</li>\n"

    html_content += f"""    </ul>

    <div class="explanation">
        <h2>Methodology Overview (LPPLS Fitting)</h2>
        <p>This analysis fits the Log-Periodic Power Law Singularity (LPPLS) model to both regular and inverse price series. The LPPLS model attempts to capture potential bubble dynamics characterized by:</p>
        <ul>
            <li><strong>Faster-than-exponential growth:</strong> Price increases accelerate according to a power law as a potential critical time approaches.</li>
            <li><strong>Log-Periodic Oscillations:</strong> Accelerating oscillations decorate the power law trend, theoretically reflecting feedback loops or heterogeneity among market participants.</li>
        </ul>
        <p>The model equation fitted to the logarithm of the price <code>log(p(t))</code> is typically represented as:</p>
        <p><code>log[p(t)] ≈ A + B*|tc - t|^m + C*|tc - t|^m * cos[ω * log|tc - t| + φ]</code></p>
        <p>Where the key parameters estimated are:</p>
        <ul>
            <li><code>tc</code>: The **critical time**, representing the model's estimate of the most likely time for the end of the bubble regime (singularity). It's NOT a precise crash prediction.</li>
            <li><code>m</code>: The **power law exponent** (typically between 0 and 1), indicating the degree of super-exponential growth.</li>
            <li><code>ω</code>: The **log-frequency** (typically between 2 and 25), indicating the frequency of oscillations on a logarithmic time scale.</li>
            <li><code>A, B, C</code>: Amplitude and scaling parameters (<code>B</code> must be negative for a bubble). (Phase <code>φ</code> is often absorbed into <code>C</code>/<code>ω</code>).</li>
        </ul>
        <p><strong>Process:</strong></p>
        <ol>
            <li>The LPPLS equation is fitted to both the log-price and log-inverse-price data within sliding windows (e.g., {LPPLS_WINDOW_SIZE} days long).</li>
            <li>This fitting process is repeated as the window slides forward through the data (e.g., every {LPPLS_STEP_SIZE} days).</li>
            <li>Each fit produces estimates for <code>tc</code>, <code>m</code>, <code>ω</code>, etc., for that specific window.</li>
            <li>Fits are filtered based on criteria (e.g., parameter ranges, proximity of <code>tc</code>) to identify potentially "confident" signals of an LPPLS regime.</li>
        </ol>
        <h3>Benefits:</h3>
        <ul>
            <li>Directly models the specific patterns proposed by Sornette for bubbles.</li>
            <li>Provides quantitative estimates of bubble parameters (<code>tc</code>, <code>m</code>, <code>ω</code>) for both regular and inverse price series.</li>
            <li>The evolution of these parameters over rolling windows can offer insights into the development or decay of potential bubble characteristics.</li>
            <li>Analyzing both regular and inverse price series can provide additional confirmation of bubble signals.</li>
        </ul>
        <h3>Possible Flaws & Limitations:</h3>
        <ul>
            <li><strong>Fitting Complexity:</strong> LPPLS fitting is a challenging non-linear optimization problem. Results can be highly sensitive to the fitting window size, starting parameters, data noise, and optimization algorithm. Fits may fail or converge to unrealistic local minima.</li>
            <li><strong>No Guaranteed Prediction:</strong> A "confident" LPPLS fit indicates the *presence of a pattern consistent with the model*. It does not guarantee a crash or predict its timing accurately. <code>tc</code> is an estimate within the model's framework and often shifts as new data arrives.</li>
            <li><strong>Parameter Interpretation:</strong> Requires expertise. Focus should be on the *stability* and *convergence* of parameters (especially <code>tc</code>) across multiple windows, and whether they fall within theoretically plausible ranges.</li>
            <li><strong>False Positives/Negatives:</strong> The model can potentially identify LPPLS patterns in data that are not true bubbles, or fail to detect patterns in noisy data.</li>
            <li><strong>Computational Cost:</strong> Rolling window fitting is computationally intensive, especially when analyzing both regular and inverse price series.</li>
            <li><strong>Data Quality:</strong> Sensitive to outliers, missing data, and the choice of price series.</li>
        </ul>
    </div>

    <div class="explanation">
        <h2>How to Read the Individual Reports</h2>
        <p>Each report shows four panels for the analyzed asset based on the rolling window LPPLS fits:</p>
        <ol>
            <li><strong>Price & Confident LPPLS Fits:</strong> The top panel displays the daily closing price. Red diamond markers (◊) indicate the end date of a rolling window where the LPPLS fit met the predefined "confidence" criteria.</li>
            <li><strong>Estimated Days to Critical Time (tc - t_end):</strong> Shows the evolution of the estimated remaining time until the critical time (<code>tc</code>) from the end of each corresponding fitting window.</li>
            <li><strong>Exponent (m):</strong> Displays the fitted power-law exponent <code>m</code> for each window.</li>
            <li><strong>Log-Frequency (ω):</strong> Shows the fitted log-frequency <code>omega</code>.</li>
        </ol>
        <p><strong>Use the interactive features:</strong> Hover over lines to see values, zoom, pan, and use the rangeslider at the bottom to focus on specific periods. Assess the consistency and plausibility of the LPPLS parameters over time, especially during periods of strong price acceleration.</p>
    </div>

</body>
</html>
"""
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"\nSuccessfully created index file: {output_filename}")
    except Exception as e:
        print(f"\nError creating index file {output_filename}: {e}")

# --- Main Execution Block (Modified) ---
if __name__ == "__main__":
    start_time = time.time()
    
    # Parse command line arguments
    args = parse_arguments()
    
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    daily_data_dict = {}
    
    if args.daily_file:
        # Process single daily file
        df, asset_name = load_daily_file(args.daily_file)
        if df is not None and not df.empty:
            daily_data_dict[asset_name] = df
        else:
            print(f"Failed to load or validate daily file '{args.daily_file}'. Exiting.")
            sys.exit(1)
    else:
        # Process all 1-minute files in data directory
        csv_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
        if not csv_files:
            print(f"Error: No CSV files found in '{DATA_DIR}'. Please add data files.")
            sys.exit(1)
            
        print(f"Found {len(csv_files)} CSV files.")
        print(f"Using up to {NUM_PROCESSES} processes for parallel analysis.")
        
        # Load and resample data
        print("\n--- Loading and Resampling Data ---")
        for f in csv_files:
            df, asset_name = load_and_resample_daily(f)
            if df is not None and not df.empty:
                daily_data_dict[asset_name] = df
            else:
                print(f"Failed to load or process data for asset '{asset_name}'. It will be excluded.")
    
    if not daily_data_dict:
        print("\nError: No data could be successfully loaded. Exiting.")
        sys.exit(1)
    
    print(f"\nSuccessfully loaded data for {len(daily_data_dict)} assets: {list(daily_data_dict.keys())}")
    
    # --- Analyze Each Asset (LPPLS) in Parallel ---
    print("\n--- Analyzing Assets via LPPLS (Parallel) ---")
    analysis_tasks = []
    for asset_name, daily_df in daily_data_dict.items():
        analysis_tasks.append((asset_name, daily_df, REPORTS_DIR))
    
    report_statuses = {}
    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.starmap(analyze_asset, analysis_tasks)
    
    for asset_name, success_status in results:
        report_statuses[asset_name] = success_status
    
    # --- Create Index HTML ---
    print("\n--- Generating Index HTML ---")
    successful_reports = {
        name: os.path.join(REPORTS_DIR, f"{name}_lppls_combined.html")
        for name, status in report_statuses.items() if status
    }
    create_index_html(successful_reports, INDEX_HTML_FILE)
    
    # --- Finish ---
    end_time = time.time()
    print(f"\nScript finished in {end_time - start_time:.2f} seconds.")
    print(f"Reports generated in '{REPORTS_DIR}'. Open '{INDEX_HTML_FILE}' to view the index.") 