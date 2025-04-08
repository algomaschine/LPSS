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

# --- Configuration ---
DATA_DIR = "data"  # Directory containing 1-minute CSV files
REPORTS_DIR = "reports_LPPLS" # Directory to save HTML reports
INDEX_HTML_FILE = 'index_LPPLS.html' # Main index file

# --- LPPLS Fitting Configuration (Original) ---
LPPLS_WINDOW_SIZE_ORIGINAL = 90  # Days: Length of the window for each fit
LPPLS_STEP_SIZE_ORIGINAL = 5     # Days: How often to perform a fit
LPPLS_BOUNDS_ORIGINAL = [
    (LPPLS_WINDOW_SIZE_ORIGINAL * 0.8, LPPLS_WINDOW_SIZE_ORIGINAL * 2.0), # tc
    (0.01, 0.99),  # m
    (2.0, 25.0),   # omega
    (None, None),  # A
    (-1.0, -1e-6), # B
    (None, None)   # C
]
TC_INITIAL_GUESS_FACTOR_ORIGINAL = 1.1

# --- LPPLS Fitting Configuration (Proper) ---
LPPLS_WINDOW_SIZE_PROPER = 60    # Days: Shorter window for more recent data
LPPLS_STEP_SIZE_PROPER = 5       # Days: Same step size for consistency
LPPLS_BOUNDS_PROPER = [
    (LPPLS_WINDOW_SIZE_PROPER * 0.8, LPPLS_WINDOW_SIZE_PROPER * 2.0), # tc
    (0.1, 0.9),   # m: Stricter bounds
    (4.0, 15.0),  # omega: Stricter bounds
    (None, None), # A
    (-1.0, -1e-6),# B
    (None, None)  # C
]
TC_INITIAL_GUESS_FACTOR_PROPER = 1.05

# --- Confidence Filtering ---
CONFIDENCE_FILTERS_ORIGINAL = {
    'max_tc_lookahead': 60,
    'm_range': (0.1, 0.9),
    'omega_range': (4, 15)
}

CONFIDENCE_FILTERS_PROPER = {
    'max_tc_lookahead': 45,      # Shorter lookahead
    'm_range': (0.2, 0.8),       # Stricter m range
    'omega_range': (5, 12),      # Stricter omega range
    'min_damping_factor': 0.1    # Additional filter for proper analysis
}

NUM_PROCESSES = max(1, cpu_count() - 3)

# --- Command Line Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(description='LPPLS Analysis Tool')
    parser.add_argument('--daily-file', type=str, help='Path to a daily CSV file for analysis')
    return parser.parse_args()

# --- Data Loading and Validation ---
def validate_daily_data(df, filepath):
    """Validates that the data is in daily format with correct column names and time intervals."""
    required_columns = ['Date', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Input file '{filepath}' is missing required columns: {missing_columns}")
        return False
    
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        print("Error: 'Date' column must be in datetime format")
        return False
    
    df = df.sort_values('Date')
    time_diffs = df['Date'].diff().dt.total_seconds()
    valid_diffs = time_diffs.between(86300, 86500)
    if not valid_diffs.all():
        print(f"Error: Time differences between rows should be approximately 24 hours")
        return False
    
    return True

def load_daily_file(filepath):
    """Loads and validates a daily CSV file."""
    try:
        print(f"Reading daily data from: {filepath}")
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        
        if not validate_daily_data(df, filepath):
            return None, None
        
        asset_name = os.path.splitext(os.path.basename(filepath))[0]
        return df, asset_name
        
    except Exception as e:
        print(f"Error loading daily file {filepath}: {e}")
        return None, None

def load_and_resample_daily(filepath):
    """Loads 1-min CSV, checks for 'date', 'close', 'volume', resamples to daily."""
    asset_name = "Unknown"
    try:
        basename = os.path.basename(filepath)
        asset_name = basename.split('_')[0]
        print(f"[{asset_name}] Reading data from: {filepath}")

        daily_cache_dir = os.path.join(DATA_DIR, "daily_cache")
        os.makedirs(daily_cache_dir, exist_ok=True)
        daily_cache_file = os.path.join(daily_cache_dir, f"{asset_name}_daily.csv")
        
        if os.path.exists(daily_cache_file):
            cache_mtime = os.path.getmtime(daily_cache_file)
            if time.time() - cache_mtime <= 3600:
                print(f"[{asset_name}] Using cached daily data")
                df_daily = pd.read_csv(daily_cache_file)
                df_daily['Date'] = pd.to_datetime(df_daily['Date'])
                return df_daily, asset_name

        try:
            df_cols = pd.read_csv(filepath, nrows=0).columns.str.lower().tolist()
            col_mapping = {col.lower(): col for col in pd.read_csv(filepath, nrows=0).columns}
        except Exception as e:
            print(f"Warning [{asset_name}]: Could not read header: {e}")
            return None, asset_name

        required_cols_lower = ['date', 'close', 'volume']
        if not all(col in df_cols for col in required_cols_lower):
            print(f"Warning [{asset_name}]: Missing required columns")
            return None, asset_name

        date_col = col_mapping['date']
        close_col = col_mapping['close']
        volume_col = col_mapping['volume']

        df_1m = pd.read_csv(filepath, usecols=[date_col, close_col, volume_col])
        df_1m.rename(columns={date_col: 'Date', close_col: 'close', volume_col: 'volume'}, inplace=True)

        df_1m['Date'] = pd.to_datetime(df_1m['Date'])
        df_1m = df_1m.dropna(subset=['Date', 'close', 'volume'])
        df_1m = df_1m[df_1m['close'] > 1e-9]
        df_1m = df_1m.sort_values('Date').set_index('Date')

        daily_agg = df_1m.resample('D', label='right', closed='right').agg({
            'close': 'last',
            'volume': 'sum'
        })
        df_daily = daily_agg.dropna(subset=['close'])
        df_daily = df_daily.reset_index()
        
        if df_daily.empty:
            print(f"Warning [{asset_name}]: No daily data after resampling")
            return None, asset_name

        df_daily.to_csv(daily_cache_file, index=False)
        return df_daily, asset_name

    except Exception as e:
        print(f"Error [{asset_name}]: {e}")
        return None, asset_name

# --- LPPLS Model and Fitting Functions ---
def lppls_model_equation(t, tc, m, omega, A, B, C):
    """Calculates log-price based on LPPLS parameters."""
    dt = tc - t
    log_dt = np.log(np.maximum(dt, 1e-9))
    power_term = np.power(np.maximum(dt, 1e-9), m)
    model_val = np.where(
        dt > 1e-9,
        A + B * power_term + C * power_term * np.cos(omega * log_dt),
        A
    )
    return model_val

def lppls_objective_function(params, t_window, log_price_window):
    """Objective function: sum of squared errors for LPPLS fit."""
    tc_rel, m, omega, A, B, C = params
    t_adjusted = t_window - t_window[0]
    tc_adjusted = tc_rel
    model_log_price = lppls_model_equation(t_adjusted, tc_adjusted, m, omega, A, B, C)
    if np.any(np.isnan(model_log_price)) or np.any(np.isinf(model_log_price)):
        return 1e12
    return np.sum((model_log_price - log_price_window)**2)

LPPLSFitResult = namedtuple("LPPLSFitResult", ["window_end_date", "success", "tc", "m", "omega", "A", "B", "C", "sse", "message", "confident"])

def fit_lppls_window(t_window, log_price_window, window_end_date, bounds, tc_initial_guess_factor):
    """Fits the LPPLS model to a single window of data with configurable parameters."""
    if len(t_window) != len(log_price_window):
        raise ValueError("Time and log-price windows must have the same length.")
    if len(t_window) < 10:
        return LPPLSFitResult(window_end_date, False, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.inf, "Window too short", False)

    tc_initial_rel = len(t_window) * tc_initial_guess_factor
    m_initial = 0.5
    omega_initial = 7.0
    try:
        poly_coeffs = np.polyfit(t_window, log_price_window, 1)
        A_initial = poly_coeffs[1]
    except:
        A_initial = np.mean(log_price_window)

    B_initial = -0.1
    C_initial = 0.0
    initial_guess = [tc_initial_rel, m_initial, omega_initial, A_initial, B_initial, C_initial]

    current_bounds = [
        (len(t_window), bounds[0][1]),
        bounds[1],
        bounds[2],
        bounds[3],
        bounds[4],
        bounds[5]
    ]

    try:
        result = minimize(
            lppls_objective_function,
            initial_guess,
            args=(t_window, log_price_window),
            method='SLSQP',
            bounds=current_bounds,
            options={'maxiter': 500, 'ftol': 1e-7, 'disp': False}
        )

        if result.success:
            tc_fit_rel, m_fit, omega_fit, A_fit, B_fit, C_fit = result.x
            sse_fit = result.fun
            tc_fit_abs_index = t_window[0] + tc_fit_rel
            return LPPLSFitResult(window_end_date, True, tc_fit_abs_index, m_fit, omega_fit, A_fit, B_fit, C_fit, sse_fit, result.message, False)
        else:
            return LPPLSFitResult(window_end_date, False, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.inf, result.message, False)

    except Exception as e:
        print(f"Error during LPPLS fit for window ending {window_end_date}: {e}")
        return LPPLSFitResult(window_end_date, False, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.inf, str(e), False)

def qualify_lppls_fit(params, window_length_points, confidence_filters):
    """Basic check if fitted parameters seem reasonable for a 'confident' bubble signal."""
    tc_rel, m, omega, A, B, C = params
    tc_days_from_win_end = tc_rel - window_length_points
    
    # Basic parameter checks
    m_ok = confidence_filters['m_range'][0] <= m <= confidence_filters['m_range'][1]
    omega_ok = confidence_filters['omega_range'][0] <= omega <= confidence_filters['omega_range'][1]
    tc_ok = 0 < tc_days_from_win_end <= confidence_filters['max_tc_lookahead']
    b_ok = B < 0
    
    # Additional damping factor check for proper analysis
    if 'min_damping_factor' in confidence_filters:
        # Use a small threshold to avoid division by very small numbers
        if omega * abs(C) > 1e-9:
            damping_factor = abs(B) / (omega * abs(C))
            damping_ok = damping_factor >= confidence_filters['min_damping_factor']
        else:
            # If C is effectively zero, consider it infinitely damped (passes check)
            damping_ok = True
    else:
        damping_ok = True
    
    return m_ok and omega_ok and tc_ok and b_ok and damping_ok

# --- Plotting and Analysis Functions ---
def generate_plots(daily_data, lppls_results_original, lppls_results_proper, output_filepath, asset_name):
    """Generates two separate HTML plots: one for original LPPLS and one for proper LPPLS."""
    try:
        if daily_data.empty:
            print(f"Warning [{asset_name}]: No daily data available for plotting.")
            return False

        print(f"[{asset_name}] Generating combined plots...")

        fig_original = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            subplot_titles=(
                f"{asset_name} - Original LPPLS Analysis",
                "Estimated Days to Critical Time (tc - t_end)",
                "Exponent (m)",
                "Log-Frequency (omega)"
            ),
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )

        fig_proper = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            subplot_titles=(
                f"{asset_name} - Proper LPPLS Analysis",
                "Estimated Days to Critical Time (tc - t_end)",
                "Exponent (m)",
                "Log-Frequency (omega)"
            ),
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )

        # Add price traces
        for fig in [fig_original, fig_proper]:
            fig.add_trace(go.Scatter(
                x=daily_data['Date'], y=daily_data['close'], mode='lines',
                name="Price", line={'color': 'blue', 'width': 1.5}
            ), row=1, col=1)

        # Process and plot results for both versions
        for fig, results, version, conf_filters, bounds in [
            (fig_original, lppls_results_original, "original", CONFIDENCE_FILTERS_ORIGINAL, LPPLS_BOUNDS_ORIGINAL),
            (fig_proper, lppls_results_proper, "proper", CONFIDENCE_FILTERS_PROPER, LPPLS_BOUNDS_PROPER)
        ]:
            if results:
                lppls_df = pd.DataFrame(results)
                lppls_df['Date'] = pd.to_datetime(lppls_df['window_end_date'])
                date_to_index = pd.Series(index=daily_data['Date'], data=np.arange(len(daily_data)))
                window_end_indices = lppls_df['Date'].map(date_to_index)
                lppls_df['tc_days_remaining'] = lppls_df['tc'] - window_end_indices

                confident_fits = lppls_df[lppls_df['confident'] == True]
                if not confident_fits.empty:
                    confident_fit_prices = daily_data.set_index('Date').loc[confident_fits['Date']]['close']
                    hover_texts = [
                        f"<b>Confident LPPLS Fit</b><br>Date: {date.strftime('%Y-%m-%d')}<br>"
                        f"Est. Days to tc: {tc_rem:.1f}<br>m: {m:.3f}<br>ω: {om:.2f}<br>Price: {price:.4f}"
                        for date, tc_rem, m, om, price in zip(
                            confident_fits['Date'], confident_fits['tc_days_remaining'],
                            confident_fits['m'], confident_fits['omega'], confident_fit_prices
                        )
                    ]
                    fig.add_trace(go.Scatter(
                        x=confident_fits['Date'], y=confident_fit_prices,
                        mode='markers', name="Confident Fit",
                        marker=dict(
                            color='red',
                            symbol='diamond',
                            size=8,
                            line=dict(width=1, color='DarkSlateGrey')
                        ),
                        text=hover_texts, hoverinfo='text'
                    ), row=1, col=1)

                # Add parameter traces
                fig.add_trace(go.Scatter(
                    x=lppls_df['Date'], y=lppls_df['tc_days_remaining'],
                    mode='lines+markers', name="Days to tc",
                    line={'color': 'green', 'width': 1}, marker={'size': 3}
                ), row=2, col=1)
                fig.add_hline(y=0, line_dash='dash', line_color='grey', line_width=1, row=2, col=1)

                fig.add_trace(go.Scatter(
                    x=lppls_df['Date'], y=lppls_df['m'],
                    mode='lines+markers', name="Exponent m",
                    line={'color': 'purple', 'width': 1}, marker={'size': 3}
                ), row=3, col=1)
                fig.add_hline(y=conf_filters['m_range'][0], line_dash='dot', line_color='grey', line_width=1, row=3, col=1)
                fig.add_hline(y=conf_filters['m_range'][1], line_dash='dot', line_color='grey', line_width=1, row=3, col=1)

                fig.add_trace(go.Scatter(
                    x=lppls_df['Date'], y=lppls_df['omega'],
                    mode='lines+markers', name="Log-Frequency ω",
                    line={'color': 'orange', 'width': 1}, marker={'size': 3}
                ), row=4, col=1)
                fig.add_hline(y=conf_filters['omega_range'][0], line_dash='dot', line_color='grey', line_width=1, row=4, col=1)
                fig.add_hline(y=conf_filters['omega_range'][1], line_dash='dot', line_color='grey', line_width=1, row=4, col=1)

        # Update layouts
        for fig, conf_filters, bounds in [
            (fig_original, CONFIDENCE_FILTERS_ORIGINAL, LPPLS_BOUNDS_ORIGINAL),
            (fig_proper, CONFIDENCE_FILTERS_PROPER, LPPLS_BOUNDS_PROPER)
        ]:
            fig.update_layout(
                height=950,
                showlegend=True,
                legend_title_text='Metrics & Fits',
                legend=dict(traceorder='reversed'),
                margin=dict(t=50, b=50)
            )
            fig.update_yaxes(title_text="Price / Fits", row=1, col=1)
            fig.update_yaxes(title_text="Days to tc", range=[-10, conf_filters['max_tc_lookahead'] * 1.5], row=2, col=1)
            fig.update_yaxes(title_text="m", range=[0, 1], row=3, col=1)
            fig.update_yaxes(title_text="ω", range=[0, bounds[2][1]*1.1], row=4, col=1)
            fig.update_xaxes(title_text="Date", row=4, col=1)
            fig.update_xaxes(showspikes=True, spikecolor="grey", spikesnap="cursor", spikemode="across", spikedash='dot', spikethickness=1)
            fig.update_yaxes(showspikes=True, spikecolor="grey", spikesnap="cursor", spikethickness=0.5)
            fig.update_xaxes(rangeslider_visible=True, row=4, col=1)

        # Save combined HTML file
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{asset_name} - Original & Proper LPPLS Analysis</title>
                <meta charset="UTF-8">
                <style>
                    .plot-container {{
                        margin-bottom: 50px;
                    }}
                    .plot-title {{
                        text-align: center;
                        font-size: 1.2em;
                        margin: 20px 0;
                    }}
                </style>
            </head>
            <body>
                <h1>{asset_name} - Original & Proper LPPLS Analysis</h1>
                <div class="plot-container">
                    <div class="plot-title">Original LPPLS Implementation</div>
                    {original_plot}
                </div>
                <div class="plot-container">
                    <div class="plot-title">Proper LPPLS Implementation</div>
                    {proper_plot}
                </div>
            </body>
            </html>
            """.format(
                asset_name=asset_name,
                original_plot=fig_original.to_html(full_html=False),
                proper_plot=fig_proper.to_html(full_html=False)
            ))

        print(f"[{asset_name}] Successfully generated combined plots: {output_filepath}")
        return True

    except Exception as e:
        print(f"Error [{asset_name}]: An error occurred during plot generation: {e}")
        return False

def analyze_asset(asset_name, daily_data, output_dir):
    """Runs both original and proper LPPLS analysis for a single asset."""
    print(f"\n--- Analyzing Asset: {asset_name} ---")
    output_filepath = os.path.join(output_dir, f"{asset_name}_analysis.html")

    if daily_data is None or daily_data.empty:
        print(f"Warning [{asset_name}]: No data available for analysis. Skipping.")
        return asset_name, False

    # Prepare data
    daily_data = daily_data.sort_values('Date').reset_index(drop=True)
    log_prices = np.log(daily_data['close'].values)
    time_indices = np.arange(len(daily_data))

    # Run both analyses
    lppls_results_original = []
    lppls_results_proper = []
    
    # Use the larger window size for the outer loop
    max_window_size = max(LPPLS_WINDOW_SIZE_ORIGINAL, LPPLS_WINDOW_SIZE_PROPER)
    step_size = LPPLS_STEP_SIZE_ORIGINAL  # Use same step size for both
    
    for i in range(max_window_size - 1, len(daily_data), step_size):
        window_end_date = daily_data['Date'].iloc[i]
        
        # Original analysis
        if i >= LPPLS_WINDOW_SIZE_ORIGINAL - 1:
            window_start_orig = i - (LPPLS_WINDOW_SIZE_ORIGINAL - 1)
            t_win_orig = time_indices[window_start_orig:i+1]
            log_p_win_orig = log_prices[window_start_orig:i+1]
            
            if np.std(log_p_win_orig) >= 1e-6:
                fit_result = fit_lppls_window(
                    t_win_orig, log_p_win_orig, window_end_date,
                    LPPLS_BOUNDS_ORIGINAL, TC_INITIAL_GUESS_FACTOR_ORIGINAL
                )
                if fit_result.success:
                    fit_result = fit_result._replace(
                        confident=qualify_lppls_fit(
                            [fit_result.tc - t_win_orig[0], fit_result.m, fit_result.omega,
                             fit_result.A, fit_result.B, fit_result.C],
                            len(t_win_orig),
                            CONFIDENCE_FILTERS_ORIGINAL
                        )
                    )
                lppls_results_original.append(fit_result)
        
        # Proper analysis
        if i >= LPPLS_WINDOW_SIZE_PROPER - 1:
            window_start_prop = i - (LPPLS_WINDOW_SIZE_PROPER - 1)
            t_win_prop = time_indices[window_start_prop:i+1]
            log_p_win_prop = log_prices[window_start_prop:i+1]
            
            if np.std(log_p_win_prop) >= 1e-6:
                fit_result = fit_lppls_window(
                    t_win_prop, log_p_win_prop, window_end_date,
                    LPPLS_BOUNDS_PROPER, TC_INITIAL_GUESS_FACTOR_PROPER
                )
                if fit_result.success:
                    fit_result = fit_result._replace(
                        confident=qualify_lppls_fit(
                            [fit_result.tc - t_win_prop[0], fit_result.m, fit_result.omega,
                             fit_result.A, fit_result.B, fit_result.C],
                            len(t_win_prop),
                            CONFIDENCE_FILTERS_PROPER
                        )
                    )
                lppls_results_proper.append(fit_result)

    # Generate plots
    plot_success = generate_plots(daily_data, lppls_results_original, lppls_results_proper, output_filepath, asset_name)
    return asset_name, plot_success

def create_index_html(report_files, output_filename):
    """Creates the index.html file with links to all reports."""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LPPLS Analysis Reports</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; line-height: 1.6; }}
        h1, h2 {{ color: #333; }}
        ul {{ list-style-type: none; padding: 0; }}
        li {{ margin-bottom: 10px; background-color: #f4f4f4; padding: 10px; border-radius: 5px; }}
        a {{ text-decoration: none; color: #007bff; font-weight: bold; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>LPPLS Analysis Reports</h1>
    <p>This page provides links to individual asset reports generated using both original and proper LPPLS implementations.</p>

    <h2>Available Reports:</h2>
    <ul>
"""
    if report_files:
        for report_name, report_path in sorted(report_files.items()):
            relative_path = os.path.join(REPORTS_DIR, os.path.basename(report_path))
            html_content += f'        <li><a href="{relative_path}">{report_name}</a></li>\n'
    else:
        html_content += "        <li>No reports were generated successfully.</li>\n"

    html_content += """    </ul>
</body>
</html>
"""
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"\nSuccessfully created index file: {output_filename}")
    except Exception as e:
        print(f"\nError creating index file {output_filename}: {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    start_time = time.time()
    args = parse_arguments()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    if args.daily_file:
        # Process single daily file
        daily_data, asset_name = load_daily_file(args.daily_file)
        if daily_data is not None:
            analyze_asset(asset_name, daily_data, REPORTS_DIR)
    else:
        # Process all files in data directory
        csv_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
        if not csv_files:
            print(f"Error: No CSV files found in '{DATA_DIR}'")
            sys.exit(1)

        print(f"Found {len(csv_files)} CSV files")
        print(f"Using up to {NUM_PROCESSES} processes for parallel analysis")

        # Load and process data
        daily_data_dict = {}
        for f in csv_files:
            df, asset_name = load_and_resample_daily(f)
            if df is not None and not df.empty:
                daily_data_dict[asset_name] = df

        if not daily_data_dict:
            print("\nError: No data could be successfully loaded")
            sys.exit(1)

        print(f"\nSuccessfully loaded daily data for {len(daily_data_dict)} assets")

        # Analyze assets in parallel
        analysis_tasks = [(name, df, REPORTS_DIR) for name, df in daily_data_dict.items()]
        report_statuses = {}
        
        with Pool(processes=NUM_PROCESSES) as pool:
            results = pool.starmap(analyze_asset, analysis_tasks)

        for asset_name, success_status in results:
            report_statuses[asset_name] = success_status

        # Create index HTML
        successful_reports = {
            name: os.path.join(REPORTS_DIR, f"{name}_analysis.html")
            for name, status in report_statuses.items() if status
        }
        create_index_html(successful_reports, INDEX_HTML_FILE)

    end_time = time.time()
    print(f"\nScript finished in {end_time - start_time:.2f} seconds")
    print(f"Reports generated in '{REPORTS_DIR}'. Open '{INDEX_HTML_FILE}' to view the index.") 