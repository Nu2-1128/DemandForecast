
import pandas as pd
import numpy as np
import streamlit as st
import warnings

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

st.title('Product Forecast Result')
st.markdown("For any questions or suggestions, please email: <a href='mailto:support@yourcompany.com'>support@yourcompany.com</a>", unsafe_allow_html=True)

# Hardcode of each file ilnks shared by developer;
# To be removed once the application is successfully hosted in Google Cloud.

public_csv_urls ={
  'time_series_analysis': 'https://drive.google.com/uc?export=download&id=1TabpcT7O-E69WDwwbAFuAWLYohAiEH1x',
  'holt_winters': 'https://drive.google.com/uc?export=download&id=12_xlfN6ckXbLFi-tgfQVfwUSjnhJ0TQo',
  'arima': 'https://drive.google.com/uc?export=download&id=1dmwzPFdsk_Y-i3Ar_IelmFduv1DTGAcM',
  'ets': 'https://drive.google.com/uc?export=download&id=1IANN4BZ4ehn5x3-XnmaWFSpL_Xg3FxKi',
  'combined': 'https://drive.google.com/uc?export=download&id=1lK2--bL7k_wV7idEiD7EaPF1wDJ66EV6',
  'melted_performance': 'https://drive.google.com/uc?export=download&id=1h0BWiat2yqe_WslgJ2xhezKRePNNOalD',
  'merged_performance': 'https://drive.google.com/uc?export=download&id=1T9to_5P-0fskcW1x3AQVTt0qQ8bCfJIR'
}

# Function to retreive file, report error if file is not found
def get_CSV_Data(file_key):
  url = public_csv_urls.get(file_key)
  if not url:
    st.error(f"Error: Invalid file key '{file_key}'.")
    return pd.DataFrame()
  try:
    # Pandas can read directly from a URL
    return pd.read_csv(url)
  except Exception as e:
    st.error(f"Error reading {file_key}: {e}")
    return pd.DataFrame()

st.set_page_config(layout="wide")

# Load the data needed for the app
time_series_analysis_result = get_CSV_Data('time_series_analysis')
holt_winter_result = get_CSV_Data('holt_winters')
arima_result = get_CSV_Data('arima')
ets_result = get_CSV_Data('ets')
combined_result = get_CSV_Data('combined')
melted_performance_analysis = get_CSV_Data('melted_performance')
merged_performance_analysis = get_CSV_Data('merged_performance')

# setup tabs name
tab_titles = ['Forecast Model Summary','Individual Items Forecast','Help']

tab1, tab2, tab3 = st.tabs(tab_titles)

with tab1:

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

with tab2:

  st.subheader('Individual Items Forecast')
  # Check if data was loaded successfully before proceeding
  if not time_series_analysis_result.empty:

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

with tab3:
  st.write("Welcome to the Product Forecast Result Application")

  st.markdown("For any questions or suggestions, please email: <a href='mailto:support@yourcompany.com'>support@yourcompany.com</a>", unsafe_allow_html=True)

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
    - Welcome to the Product Forecast Result Application
    - For any questions or suggestions, please email: <a href='mailto:support@yourcompany.com'>support@yourcompany.com</a>
  """)
