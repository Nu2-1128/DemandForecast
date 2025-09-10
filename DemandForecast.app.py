
import os
import pandas as pd
import numpy as np
import streamlit as st
import warnings
import io # Import the io module

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

# Hardcode of each file ilnks shared by developer;
# To be removed once the application is successfully hosted in Google Cloud.

#public_csv_urls ={
#  'time_series_analysis': 'https://drive.google.com/uc?export=download&id=1TabpcT7O-E69WDwwbAFuAWLYohAiEH1x',
#  'holt_winters': 'https://drive.google.com/uc?export=download&id=12_xlfN6ckXbLFi-tgfQVfwUSjnhJ0TQo',
#  'arima': 'https://drive.google.com/uc?export=download&id=1dmwzPFdsk_Y-i3Ar_IelmFduv1DTGAcM',
#  'ets': '1IANN4BZ4ehn5x3-XnmaWFSpL_Xg3FxKi',
#  'combined': '1lK2--bL7k_wV7idEiD7EaPF1wDJ66EV6',
#  'melted_performance': '1h0BWiat2yqe_WslgJ2xhezKRePNNOalD',
#  'merged_performance': '1T9to_5P-0fskcW1x3AQVTt0qQ8bCfJIR'
#}

public_csv_ids ={
    'auto_arima_results': '1S7CqfAH_lnKcMb_FIVXfAApdDPE9a1O2',
    'baseline_results': '1_OyODMabWzq92J5G6VLmrxldN4wRjQwA',
    'combined_forecasts_df': '16Ak2oVyUemlexY1EEkshzbO_2bs5Nl8D',
    'comany_data_treated': '1cJHUNHx6rxocFMmIXgGGI3jjEVg50dSq',
    'company_data': '1bHd4Rc3t6lPLM2OY931RoTJmMHQF-Hmu',
    'ets_results': '1hk1unwWuPYwhjLH6dDvyPd5wUxLvi_rJ', 
    'holt_winters_results': '1XIoeToA9JaPg_Z8Fo5i2d17DF0R3nBcK',
    'hybrid_df': '1J2YUWbCsBwi-POwdDqu7CId8Zpf2E6ST',
    'selected_items_df': '10rumhJAe93ThQxgpLPEY6Y1SsD_yy7G3',
    'shap_values_df': '10xbgC0M3iNDUEnfbyxQaa32aOiQ8OR95',
    'time_series_analysis_df': '1whpsUs-3ULPsg8WnlT1vMBvL99uTtHZo',
    'truncated_company_data': '1cFkzpSSqAeb81mYusNSHWtgPBWzBo67E'
}
                 

#  'time_series_analysis': '1TabpcT7O-E69WDwwbAFuAWLYohAiEH1x',
#  'holt_winters': '12_xlfN6ckXbLFi-tgfQVfwUSjnhJ0TQo',
#  'arima': '1dmwzPFdsk_Y-i3Ar_IelmFduv1DTGAcM',
#  'ets': '1IANN4BZ4ehn5x3-XnmaWFSpL_Xg3FxKi',
#  'combined': '1lK2--bL7k_wV7idEiD7EaPF1wDJ66EV6',
#  'melted_performance': '1h0BWiat2yqe_WslgJ2xhezKRePNNOalD',
#  'merged_performance': '1T9to_5P-0fskcW1x3AQVTt0qQ8bCfJIR'


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
holt_winter_result = get_DataFrame_from_File('holt_winters_results')
hybrid_result = get_DataFrame_from_File('hybrid_df')
selected_items_df = get_DataFrame_from_File('selected_items_df')
shap_values_result = get_DataFrame_from_File('shap_values_df')
time_series_analysis_result = get_DataFrame_from_File('time_series_analysis_df')
truncated_company_data = get_DataFrame_from_File('truncated_company_data')

#time_series_analysis_result = get_DataFrame_from_File('time_series_analysis')
#holt_winter_result = get_DataFrame_from_File('holt_winters')
#arima_result = get_DataFrame_from_File('arima')
#ets_result = get_DataFrame_from_File('ets')
#combined_result = get_DataFrame_from_File('combined')
#melted_performance_analysis = get_DataFrame_from_File('melted_performance')
#merged_performance_analysis = get_DataFrame_from_File('merged_performance')


# setup tabs name
tab_titles = ['Forecast Model Summary','Individual Items Forecast','Help']

tab1, tab2, tab3 = st.tabs(tab_titles)

with tab1:

st.subheader("Algorithm summary")

st.write('Auto ARIMA:')
st.dataframe(auto_arima_result)

st.write('BASELINE:')
st.dataframe(baseline_result)

"""
  # Check if data was loaded successfully before proceeding
  if not melted_performance_analysis.empty:
    performance_metrics_df = melted_performance_analysis[melted_performance_analysis['Metric_Type'].isin(['RMSE','SMAPE','MAE','Duration'].copy())]

    performance_metrics_df['Trend & Seasonality'] = performance_metrics_df.apply(
        lambda row: f"Trend: {row['has_trend']}, Seasonality: {row['has_seasonality']}", axis=1
    )

    # Initialization. Each container include a collection of visualization to be displayed
    Metric_container = st.container(border = True)
    HW_result_Container = st.container(border = True)
    ARIMA_result_Container = st.container(border = True)
    ETS_result_Container = st.container(border = True)

    with Metric_container:
      st.subheader('Forecast Model Summary')
      st.write('Performance metrics analysis - Review RMSE, SMAPE, MAE and Duration between the three models:')

      st.write('RMSE - Root Mean Square Error. metrics the square root of the average of the squared differences between predicted and actual values. The lower the closer to the original value.')
      st.write('SMAPE - Symmetric Mean Absolute Percentage Error. A percentage-based error metric that measures the accuracy of a forecast by taking the average of the absolute percentage errors. A lower SMAPE value indicates a more accurate forecast.')
      st.write('MAE - Mean Absolute Error. A percentage-based error metric that measures the accuracy of a forecast by taking the average of the absolute differences between predicted and actual values. A lower MAE value indicates a more accurate forecast')
      st.write('Duration - Time (in seconds) for processing by the model')

      g = sns.catplot(
          data=performance_metrics_df,
          x='Model',
          y='Value',
          hue='Trend & Seasonality',
          col='Metric_Type',
          kind='bar',
          sharey=False, # Allow different y-axis scales for different metrics
          height=6,
          aspect=0.8,
          # special setup for colorblind
          palette='colorblind'
      )

      st.pyplot(g.fig)

    # Check if data was loaded successfully before proceeding
    if not holt_winter_result.empty:
        with HW_result_Container:
          st.subheader('Holt Winters Analysis')

          col1, col2, col3, col4 = st.columns(4)

          with col1:
            plt.figure()
            st.write('Distribution of RMSE (HW)')
            sns.histplot(holt_winter_result['RMSE'], kde=True, bins=10)
            st.pyplot(plt, use_container_width=True)
          with col2:
            plt.figure()
            st.write('Distribution of SMAPE (HW)')
            sns.histplot(holt_winter_result['SMAPE'], kde=True, bins=10)
            st.pyplot(plt, use_container_width=True)
          with col3:
            plt.figure()
            st.write('Distribution of MAE (HW)')
            sns.histplot(holt_winter_result['MAE'], kde=True, bins=10)
            st.pyplot(plt, use_container_width=True)
          with col4:
            plt.figure()
            st.write('Distribution of Duration (HW)')
            sns.histplot(holt_winter_result['Duration'], kde=True, bins=10)
            st.pyplot(plt, use_container_width=True)

    # Check if data was loaded successfully before proceeding
    if not arima_result.empty:
        with ARIMA_result_Container:
          st.subheader('ARIMA Analysis')

          col1, col2, col3, col4 = st.columns(4)

          with col1:
            plt.figure()
            st.write('Distribution of RMSE (ARIMA)')
            sns.histplot(arima_result['RMSE'], kde=True, bins=10)
            st.pyplot(plt, use_container_width=True)
          with col2:
            plt.figure()
            st.write('Distribution of SMAPE (ARIMA)')
            sns.histplot(arima_result['SMAPE'], kde=True, bins=10)
            st.pyplot(plt, use_container_width=True)
          with col3:
            plt.figure()
            st.write('Distribution of MAE (ARIMA)')
            sns.histplot(arima_result['MAE'], kde=True, bins=10)
            st.pyplot(plt, use_container_width=True)
          with col4:
            plt.figure()
            st.write('Distribution of Duration (ARIMA)')
            sns.histplot(arima_result['Duration'], kde=True, bins=10)
            st.pyplot(plt, use_container_width=True)

    # Check if data was loaded successfully before proceeding
    if not ets_result.empty:
        with ETS_result_Container:
          st.subheader('ETS Analysis')

          col1, col2, col3, col4 = st.columns(4)
          with col1:
            plt.figure()
            st.write('Distribution of RMSE (ETS)')
            sns.histplot(ets_result['RMSE'], kde=True, bins=10)
            st.pyplot(plt, use_container_width=True)
          with col2:
            plt.figure()
            st.write('Distribution of SMAPE (ETS)')
            sns.histplot(ets_result['SMAPE'], kde=True, bins=10)
            st.pyplot(plt, use_container_width=True)
          with col3:
            plt.figure()
            st.write('Distribution of MAE (ETS)')
            sns.histplot(ets_result['MAE'], kde=True, bins=10)
            st.pyplot(plt, use_container_width=True)
          with col4:
            plt.figure()
            st.write('Distribution of Duration (ETS)')
            sns.histplot(ets_result['Duration'], kde=True, bins=10)
            st.pyplot(plt, use_container_width=True)

  else:
      st.warning("Could not load the necessary data for the Forecast Model Summary tab.")

"""
with tab2:

  st.subheader('Individual Items Forecast')

"""
  # Check if data was loaded successfully before proceeding
  if not time_series_analysis_result.empty and not merged_performance_analysis.empty and not combined_result.empty and not arima_result.empty and not ets_result.empty and not holt_winter_result.empty:

      # Select dropdown list
      prod_list = time_series_analysis_result['Item'].unique().tolist()

      st.write('Please select an Item from the dropdown list:')
      selected_prod = st.selectbox('', prod_list)

      # Initialize container
      Stat_container = st.container(border=True)
      Performance_container = st.container(border=True)
      Graph_container = st.container(border=True)
      ARIMA_container = st.container(border=True)
      ETS_container = st.container(border=True)
      HW_Container = st.container(border=True)

      with Stat_container:
        if (selected_prod != ''):
          st.subheader('Item test for trend and seasonality')
          filtere_df = time_series_analysis_result[time_series_analysis_result['Item'] == selected_prod]
          st.write(filtere_df)
      with Performance_container:
        if (selected_prod != ''):
          st.subheader('Performance Analysis')
          st.write(merged_performance_analysis[merged_performance_analysis['Item'] == selected_prod])
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
      with ARIMA_container:
        if (selected_prod != ''):
          st.subheader('ARIMA Analysis')
          st.write(arima_result[arima_result['Item'] == selected_prod])
      with ETS_container:
        if (selected_prod != ''):
          st.subheader('ETS Analysis')
          st.write(ets_result[ets_result['Item'] == selected_prod])
      with HW_Container:
        if (selected_prod != ''):
          st.subheader('Holt Winters Analysis')
          st.write(holt_winter_result[holt_winter_result['Item'] == selected_prod])

  else:
      st.warning("Could not load the necessary data to run the application.")
"""

with tab3:

  st.markdown("""
  Product Forecast Result is designed as a web-based applicaiton to allow users to review an exercise conducted by the company developer for evaluating different forecast algorithms.

  It utilized the company customer demand data, with basic data cleaning include handling NAN value and Outliers, forecast with the optimal parameters using Holt-Winters, ARIMA, and ETS algorithm.

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
