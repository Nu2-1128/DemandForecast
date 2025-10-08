
import os
import pandas as pd
import numpy as np
import streamlit as st
import warnings
import io # Import the io module
import shap

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
# Import service_account from google.oauth2.service_account
from google.oauth2 import service_account
from googleapiclient.discovery import build
import google.auth # Import the google.auth module

warnings.filterwarnings('ignore')

st.title('Product Forecast Result')
st.markdown("For any questions or suggestions, please email: <a href='mailto:support@yourcompany.com'>support@yourcompany.com</a>", unsafe_allow_html=True)

public_csv_ids ={
    'auto_arima_results': '1S7CqfAH_lnKcMb_FIVXfAApdDPE9a1O2',
    'baseline_results': '1_OyODMabWzq92J5G6VLmrxldN4wRjQwA',
    'combined_forecasts_df': '16Ak2oVyUemlexY1EEkshzbO_2bs5Nl8D',
    'company_data_treated': '1cJHUNHx6rxocFMmIXgGGI3jjEVg50dSq',
    'company_data': '1bHd4Rc3t6lPLM2OY931RoTJmMHQF-Hmu',
    'ets_results': '1hk1unwWuPYwhjLH6dDvyPd5wUxLvi_rJ',
    'holt_winters_results': '1XIoeToA9JaPg_Z8Fo5i2d17DF0R3nBcK',
    'hybrid_df': '1J2YUWbCsBwi-POwdDqu7CId8Zpf2E6ST',
    'selected_items_df': '10rumhJAe93ThQxgpLPEY6Y1SsD_yy7G3',
    'shap_values_df': '10xbgC0M3iNDUEnfbyxQaa32aOiQ8OR95',
    'time_series_analysis_df': '1whpsUs-3ULPsg8WnlT1vMBvL99uTtHZo',
    'truncated_company_data': '1cFkzpSSqAeb81mYusNSHWtgPBWzBo67E',
    'best_model': '1rCKQ2FBWVkw_zyhF_8T5jn27k_D8F5cu'
}

# Function to download file from Google Drive
def download_file(file_key):
  # Try to load credentials from environment variable or default
  try:
      credentials = None
      # If deploying, load credentials from a file specified by an environment variable
      if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
          credentials_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
          credentials = service_account.Credentials.from_service_account_file(credentials_path)
      else:
          # In Colab or other environments, use default credentials
          credentials, project = google.auth.default()

      if credentials is None:
        st.error("Google Cloud credentials not found. Please set GOOGLE_APPLICATION_CREDENTIALS environment variable or configure default credentials.")
        return None


      scoped_credentials = credentials.with_scopes(
          ['https://www.googleapis.com/auth/drive.readonly'])

      service = build('drive', 'v3', credentials=scoped_credentials)

      file_id = public_csv_ids.get(file_key)
      if not file_id:
        st.error(f"Error: Invalid file key '{file_key}'.")
        return None # Return None instead of an empty DataFrame for download
      request = service.files().get_media(fileId=file_id)
      response = request.execute()

      # Return the response content, not a DataFrame
      return response

  except Exception as e:
      st.error(f"Error downloading {file_key}: {e}")
      return None

# Function to retreive file and read into pandas DataFrame
def get_DataFrame_from_File(file_key):
  file_content = download_file(file_key)
  if file_content:
    try:
      # Read the file content into a pandas DataFrame
      return pd.read_csv(io.BytesIO(file_content))
    except Exception as e:
      st.error(f"Error reading {file_key} into DataFrame: {e}")
      return pd.DataFrame()
  else:
    return pd.DataFrame()


st.set_page_config(layout="wide")

# Load the data needed for the app
# Use get_DataFrame_from_File for reading from downloaded files

auto_arima_result = get_DataFrame_from_File('auto_arima_results')
baseline_result = get_DataFrame_from_File('baseline_results')
combined_result = get_DataFrame_from_File('combined_forecasts_df')
company_data_treated = get_DataFrame_from_File('company_data_treated')
company_data = get_DataFrame_from_File('company_data')
ets_result = get_DataFrame_from_File('ets_results')
holt_winters_result = get_DataFrame_from_File('holt_winters_results')
hybrid_result = get_DataFrame_from_File('hybrid_df')
selected_items_df = get_DataFrame_from_File('selected_items_df')
shap_values_result = get_DataFrame_from_File('shap_values_df')
time_series_analysis_result = get_DataFrame_from_File('time_series_analysis_df')
truncated_company_data = get_DataFrame_from_File('truncated_company_data')
best_model = get_DataFrame_from_File('best_model')

# setup tabs name
tab_titles = ['Forecast Model Summary','Individual Items Forecast','Help']

tab1, tab2, tab3 = st.tabs(tab_titles)

with tab1:

  st.subheader("Algorithm summary")

  # Initialization. For each Container to be used
  Metric_container = st.container(border = True)
  Best_container = st.container(border = True)
  SHAP_container = st.container(border = True)
  RMSE_Container = st.container(border = True)

  rmse_data = pd.DataFrame({
    'ETS': ets_result['RMSE'],
    'Holt-Winters': holt_winters_result['RMSE'],
    'AutoARIMA': auto_arima_result['RMSE'],
    'Hybrid': hybrid_result['RMSE'],
    'Baseline': baseline_result['RMSE']
  })

  smape_data = pd.DataFrame({
      'ETS': ets_result['SMAPE'],
      'Holt-Winters': holt_winters_result['SMAPE'],
      'AutoARIMA': auto_arima_result['SMAPE'],
      'Hybrid': hybrid_result['SMAPE'],
      'Baseline': baseline_result['SMAPE']
  })

  mae_data = pd.DataFrame({
      'ETS': ets_result['MAE'],
      'Holt-Winters': holt_winters_result['MAE'],
      'AutoARIMA': auto_arima_result['MAE'],
      'Hybrid': hybrid_result['MAE'],
      'Baseline': baseline_result['MAE']
  })

  duration_data = pd.DataFrame({
      'ETS': ets_result['Duration'],
      'Holt-Winters': holt_winters_result['Duration'],
      'AutoARIMA': auto_arima_result['Duration'],
      'Hybrid': hybrid_result['Duration'],
      'Baseline': baseline_result['Duration']
  })

  if rmse_data.empty or smape_data.empty or mae_data.empty or duration_data.empty:
    st.warning("⚠️ One or more of the metric dataframes is empty. Cannot generate plots.")
    st.info("Please ensure your data files are not empty and are loaded correctly.")
  else:

    with Metric_container:

      st.subheader('Forecast Model Summary')
      st.write('Performance metrics analysis - Review RMSE, SMAPE, MAE and Duration between the five models:')

      # Create subplots for box plots for all four metrics
      fig, axes = plt.subplots(2, 2, figsize=(12, 8))
      fig.suptitle('Comparison of Model Metrics', fontsize=16)

      # Box plot for RMSE
      sns.boxplot(data=rmse_data, ax=axes[0, 0], palette="viridis")
      axes[0, 0].set_title('RMSE Distribution by Model')
      axes[0, 0].set_ylabel('RMSE')
      axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
      axes[0, 0].tick_params(axis='x', rotation=15)

      # Box plot for SMAPE
      sns.boxplot(data=smape_data, ax=axes[0, 1], palette="plasma")
      axes[0, 1].set_title('SMAPE Distribution by Model')
      axes[0, 1].set_ylabel('SMAPE')
      axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
      axes[0, 1].tick_params(axis='x', rotation=15)

      # Box plot for MAE
      sns.boxplot(data=mae_data, ax=axes[1, 0], palette="magma")
      axes[1, 0].set_title('MAE Distribution by Model')
      axes[1, 0].set_ylabel('MAE')
      axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
      axes[1, 0].tick_params(axis='x', rotation=15)

      # Box plot for Duration
      sns.boxplot(data=duration_data, ax=axes[1, 1], palette="cividis")
      axes[1, 1].set_title('Training Duration by Model')
      axes[1, 1].set_ylabel('Duration (seconds)')
      axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
      axes[1, 1].tick_params(axis='x', rotation=15)

      plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

      st.pyplot(fig)

    with Best_container:

      st.subheader('Number of items per best model by metric')
      st.write('From each metric, the best model identifed by each item, and count of the item. To consider the best preferred model.')

      metrics = ['RMSE','SMAPE','MAE']
      counts_cols = [f'Count_Lowest_{metric}' for metric in metrics]

      fig_best, axes_best = plt.subplots(1, len(metrics), figsize=(15, 6))

      for i, metric in enumerate(metrics):
          sns.barplot(data=best_model, y='Series_Type', x=f'Count_Lowest_{metric}', ax=axes_best[i], palette='viridis_r') # Changed palette to viridis_r for reversed colors
          axes_best[i].set_title(f'Number of Items by {metric} for best model')
          axes_best[i].set_xlabel('Number of Items')
          axes_best[i].set_ylabel('Model Type')
          axes_best[i].grid(axis='x', alpha=0.75)

      plt.tight_layout()
      st.pyplot(fig_best)
    
    with SHAP_container:

      st.subhearder('SHAP for XGBoost process')
      st.write('For the Hybrid model - XGBoost process, to evalulate each factor effectiveness as a predictor')

      feature_names = ['month', 'year', 'lag_1_qty', 'rolling_3m_avg_qty', 'Promo']
      shap_values = shap_values_result[feature_names].values
      X_values = shap_values_result[feature_names]

      fig_shap, ax_shap = plt.subplots()
      shap.summary_plot(shap_values, X_values , feature_names=feature_names)
      st.pyplot(fig_shap)

    with RMSE_Container:

      st.subheader('Combined RMSE Distribution')
      st.write('Combined display of the RMSE distribution for all models')

      # Determine global x-limits for a consistent view across histograms
      global_rmse_min = rmse_data.min().min()
      global_rmse_max = rmse_data.max().max()

      # Create the figure for the histogram
      fig_hist, ax_hist = plt.subplots(figsize=(12, 7))

      # Plot overlaid histograms for each model's RMSE
      colors = {'ETS': 'skyblue', 'Holt-Winters': 'lightcoral', 'AutoARIMA': 'lightgreen', 'Hybrid': 'gold', 'Baseline': 'purple'}
      for model in rmse_data.columns:
          ax_hist.hist(rmse_data[model], bins=20, density=True, alpha=0.75, label=model, color=colors.get(model, 'gray'))

      ax_hist.set_title('Combined RMSE Distribution of Forecasting Models')
      ax_hist.set_xlabel('RMSE')
      ax_hist.set_ylabel('Density')
      ax_hist.set_xlim([global_rmse_min, global_rmse_max])
      ax_hist.legend()
      ax_hist.grid(axis='y', linestyle='--', alpha=0.75)

      plt.tight_layout()
      st.pyplot(fig_hist)

with tab2:

  st.subheader('Individual Items Forecast')

  # Check if data was loaded successfully before proceeding
  if not time_series_analysis_result.empty and not combined_result.empty and not auto_arima_result.empty and not ets_result.empty and not holt_winters_result.empty and not hybrid_result.empty and not baseline_result.empty:

      # Select dropdown list
      prod_list = time_series_analysis_result['Item'].unique().tolist()

      st.write('Please select an Item from the dropdown list:')
      selected_prod = st.selectbox('', prod_list)

      # Initialize container
      Stat_container = st.container(border=True)
      Graph_container = st.container(border=True)
      BASELINE_Container = st.container(border=True)
      ARIMA_container = st.container(border=True)
      ETS_container = st.container(border=True)
      HW_Container = st.container(border=True)
      HYBRID_Container = st.container(border=True)


      with Stat_container:
        if (selected_prod != ''):
          st.subheader('Item test for trend and seasonality')
          filtere_df = time_series_analysis_result[time_series_analysis_result['Item'] == selected_prod]
          st.write(filtere_df)
      with Graph_container:
        if (selected_prod != ''):
          st.subheader('Time Series Analysis with forecast')
          #st.line_chart(combined_result[combined_result['Item'] == selected_prod], x='PDay',y='Value',color='Series_Type')
          # special setup for color blind
          color_map = {'Actual': 'blue', 'ARIMA':'orange','ETS':'green','Holt_Winters':'red'}
          fig = px.line(
              combined_result[combined_result['Item'] == selected_prod],
              x='PDay',
              y='Value',
              color='Series_Type',
              markers=True,
              line_dash='Series_Type',
              color_discrete_map=color_map
              )
          fig.update_layout(
              title=f'Time Series Analysis for {selected_prod}')
          st.plotly_chart(fig, use_container_width=True)
      with BASELINE_Container:
        if (selected_prod != ''):
          st.subheader('Baseline Analysis')
          st.write(baseline_result[baseline_result['Item'] == selected_prod])
      with ARIMA_container:
        if (selected_prod != ''):
          st.subheader('ARIMA Analysis')
          st.write(auto_arima_result[auto_arima_result['Item'] == selected_prod])
      with ETS_container:
        if (selected_prod != ''):
          st.subheader('ETS Analysis')
          st.write(ets_result[ets_result['Item'] == selected_prod])
      with HW_Container:
        if (selected_prod != ''):
          st.subheader('Holt Winters Analysis')
          st.write(holt_winters_result[holt_winters_result['Item'] == selected_prod])
      with HYBRID_Container:
        if (selected_prod != ''):
          st.subheader('Hybrid Analysis')
          st.write(hybrid_result[hybrid_result['Item'] == selected_prod])

      # Add download button
      if selected_prod != '':
        # Combine all dataframes for the selected product into a single dataframe for download
        download_df = pd.concat([
            time_series_analysis_result[time_series_analysis_result['Item'] == selected_prod].assign(Source='TimeSeriesAnalysis'),
            combined_result[combined_result['Item'] == selected_prod].assign(Source='CombinedForecasts'),
            baseline_result[baseline_result['Item'] == selected_prod].assign(Source='Baseline'),
            auto_arima_result[auto_arima_result['Item'] == selected_prod].assign(Source='ARIMA'),
            ets_result[ets_result['Item'] == selected_prod].assign(Source='ETS'),
            holt_winters_result[holt_winters_result['Item'] == selected_prod].assign(Source='HoltWinters'),
            hybrid_result[hybrid_result['Item'] == selected_prod].assign(Source='Hybrid')
        ], ignore_index=True)

        csv = download_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data for Selected Item as CSV",
            data=csv,
            file_name=f'{selected_prod}_forecast_data.csv',
            mime='text/csv',
        )

  else:
      st.warning("Could not load the necessary data to run the application.")

with tab3:

  st.markdown("""
  Product Forecast Result is designed as a web-based applicaiton to allow users to review an exercise conducted by the company developer for evaluating different forecast algorithms.

  It utilized the company customer demand data, with basic data cleaning include handling NAN value and Outliers, forecast with the optimal parameters using Holt-Winters, ARIMA, ETS, Hybrid approach of ARIMA+XGBoost, and a baseline algorithm.

  This application serves with a summary analysis of the forecast result, and a page for user to select individual items to review each corresponding forecast performances.

  Below is what you can expect to find in each tab:

  - **Forecast Model Summary**
    - Performance metrics analysis - Review RMSE, SMAPE, MAE and Duration between the three models
    - Distribution of RMSE (HW), SMAPE (HW), MAE (HW), and Duration (HW)
    - Distribution of RMSE (ARIMA), SMAPE (ARIMA), MAE (ARIMA), and Duration (ARIMA)
    - Distribution of RMSE (ETS), SMAPE (ETS), MAE (ETS), and Duration (ETS)

  - **Individual Items Forecast**
    - Please select an Item from the dropdown list:
    - Item test for trend and seasonality
    - Performance Analysis
    - Time Series Analysis with forecast
    - ARIMA Analysis
    - ETS Analysis
    - Holt Winters Analysis

  - **Help**
    - Information about the application design concept and background
    - Contact information
    - About the developer
      - Project is created and design by Andy Tang for education use.
  """)
