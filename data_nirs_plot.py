import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import requests
from io import BytesIO

# Your Dropbox app credentials
APP_KEY = 'cn9p3vo0x35zli6'
APP_SECRET = 'jyans6nturbyufu'
REFRESH_TOKEN = 'yCBOviYIjVcAAAAAAAAAAYgNuht-Na_pJ91-hEO9CfeQdLLjkUnQt7DTK3ZMTETY'

def refresh_access_token():
    url = "https://api.dropbox.com/oauth2/token"
    data = {
        "grant_type": "refresh_token",
        "refresh_token": REFRESH_TOKEN,
        "client_id": APP_KEY,
        "client_secret": APP_SECRET
    }
    response = requests.post(url, data=data)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        st.error(f"Failed to refresh access token: {response.text}")
        st.stop()

def download_file_from_dropbox(path, access_token):
    url = f"https://content.dropboxapi.com/2/files/download"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Dropbox-API-Arg": f"{{\"path\": \"{path}\"}}"
    }
    response = requests.post(url, headers=headers)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        st.error(f"Failed to download file: {response.text}")
        st.stop()

# Define interventions and variables
interventions = ['post', 'pre']
variables = ['SmO2', 'THb', 'TSI']

# Refresh access token
access_token = refresh_access_token()

# Load data from Dropbox
data = {}
dropbox_base_path = '/UniBern/UniBern PhD/Publications/Pre-Post NIRS scale project/Data clean/'
for intervention in interventions:
    for variable in variables:
        filename = f'{dropbox_base_path}masterTable_{intervention}_{variable}.xlsx'
        file_stream = download_file_from_dropbox(filename, access_token)
        df = pd.read_excel(file_stream)
        if intervention in ['pre', 'post']:
            df = df.iloc[:600]  # Limit to 600 seconds for pre and post interventions
        data[(intervention, variable)] = df

# Load additional participant information
info_path = 'C:/Users/andri/Dropbox (Personal)/UniBern/UniBern PhD/Publications/Pre-Post NIRS scale project/Data clean/analysis_master table.xlsx'
info_df = pd.read_excel(info_path, sheet_name='pre_post')

# Extract relevant columns
participant_info = info_df.iloc[:, [0, 1, 8, 11, 17]]
participant_info.columns = ['Participant', 'Sex', 'Fitzpatrick Skintype', 'Fitness', 'Moxy Placement']

# Sidebar filters
sex_options = participant_info['Sex'].unique().tolist()
fitzpatrick_options = participant_info['Fitzpatrick Skintype'].unique().tolist()
fitness_options = participant_info['Fitness'].unique().tolist()
moxy_placement_options = participant_info['Moxy Placement'].unique().tolist()

selected_sex = st.sidebar.multiselect('Select Sex', sex_options, default=sex_options)
selected_fitzpatrick = st.sidebar.multiselect('Select Fitzpatrick Skintype', fitzpatrick_options, default=fitzpatrick_options)
selected_fitness = st.sidebar.multiselect('Select Fitness', fitness_options, default=fitness_options)
selected_moxy_placement = st.sidebar.multiselect('Select Moxy Placement', moxy_placement_options, default=moxy_placement_options)

# Filter participants based on selected criteria
filtered_participants = participant_info[
    (participant_info['Sex'].isin(selected_sex)) &
    (participant_info['Fitzpatrick Skintype'].isin(selected_fitzpatrick)) &
    (participant_info['Fitness'].isin(selected_fitness)) &
    (participant_info['Moxy Placement'].isin(selected_moxy_placement))
]['Participant']

# Calculate SmO2 min, SmO2 max, TSI min, TSI max for each participant
def calculate_min_max(df):
    min_values = df.iloc[60:].rolling(window=10).mean().min()
    max_values = df.iloc[60:].rolling(window=10).mean().max()
    return min_values, max_values

derived_data = {}
participant_codes = data[('pre', 'SmO2')].columns  # Assuming all tables have the same participant codes
for intervention in interventions:
    for variable in ['SmO2', 'TSI']:
        df = data[(intervention, variable)]
        min_values, max_values = calculate_min_max(df)
        derived_data[(intervention, f'{variable} min')] = min_values.reindex(participant_codes)
        derived_data[(intervention, f'{variable} max')] = max_values.reindex(participant_codes)

# Combine original and derived variables
all_combos = [f"{intervention} {variable}" for intervention in interventions for variable in variables]
derived_combos = [f"{intervention} {variable}" for intervention in interventions for variable in ['SmO2 min', 'SmO2 max', 'TSI min', 'TSI max']]
combo_options = all_combos + derived_combos

# Filter data based on selected participants
def filter_data(df, participants):
    return df[participants]

for key in data.keys():
    data[key] = filter_data(data[key], filtered_participants)

for key in derived_data.keys():
    derived_data[key] = derived_data[key].reindex(filtered_participants)

# Selection for time series plot
selected_combos = st.sidebar.multiselect('Select Interventions and Variables for Time Series Plot', all_combos, default=all_combos)

# Plot setup for time series
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# Plot data for time series
for combo in selected_combos:
    intervention, variable = combo.split()
    selected_data = data[(intervention, variable)]
    mean_values = selected_data.mean(axis=1)
    ci95 = stats.sem(selected_data, axis=1) * stats.t.ppf((1 + 0.95) / 2., selected_data.shape[1] - 1)
    
    time_points = range(len(mean_values))
    
    if variable in ['SmO2', 'TSI']:
        ax1.plot(time_points, mean_values, label=f'{intervention} - {variable}')
        ax1.fill_between(time_points, mean_values - ci95, mean_values + ci95, alpha=0.3)
    elif variable == 'THb':
        ax2.plot(time_points, mean_values, label=f'{intervention} - {variable}', linestyle='--')
        ax2.fill_between(time_points, mean_values - ci95, mean_values + ci95, alpha=0.3)

# Axes labels and limits
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Percentage (%)')
ax1.set_ylim(0, 100)

ax2.set_ylabel('Arbitrary Units (AU)')
ax2.set_ylim(10, 14)

# Title and legends
ax1.set_title('Mean Values over Time for Selected Interventions and Variables with 95% CI')
fig.legend(loc='upper right')

# Display the time series plot
st.pyplot(fig)

# Display derived data table
st.write("### Derived Data for Each Participant")
table_data = pd.DataFrame(index=participant_codes)
for (intervention, variable), values in derived_data.items():
    table_data[f'{intervention} {variable}'] = values

st.dataframe(table_data)

# Selection for Bland-Altman plot
st.sidebar.write("Bland-Altman Plot Selection")
ba_selected_combos = st.sidebar.multiselect('Select Two Interventions and Variables for Bland-Altman Plot', derived_combos)

if len(ba_selected_combos) == 2:
    # Extract selected interventions and variables
    combo1 = ba_selected_combos[0].split()
    combo2 = ba_selected_combos[1].split()
    intervention1, variable1 = combo1[0], ' '.join(combo1[1:])
    intervention2, variable2 = combo2[0], ' '.join(combo2[1:])
    
    # Ensure the selections are valid
    if (intervention1, variable1) in derived_data and (intervention2, variable2) in derived_data:
        ba_data1 = derived_data[(intervention1, variable1)]
        ba_data2 = derived_data[(intervention2, variable2)]

        # Calculate mean and difference
        mean_values = (ba_data1 + ba_data2) / 2
        diff_values = ba_data1 - ba_data2

        # Plot Bland-Altman
        fig, ax = plt.subplots()
        ax.scatter(mean_values, diff_values, label='Data points')
        ax.axhline(diff_values.mean(), color='gray', linestyle='--', label='Mean difference')
        ax.axhline(diff_values.mean() + 1.96 * diff_values.std(), color='red', linestyle='--', label='Upper 95% limit')
        ax.axhline(diff_values.mean() - 1.96 * diff_values.std(), color='blue', linestyle='--', label='Lower 95% limit')

        # Axes labels and title
        ax.set_xlabel('Mean of Two Measurements')
        ax.set_ylabel('Difference Between Measurements')
        ax.set_title('Bland-Altman Plot')
        ax.legend()

        # Display the Bland-Altman plot
        st.pyplot(fig)
    else:
        st.error("Please select valid intervention and variable combinations.")
else:
    st.warning("Please select exactly two combinations for Bland-Altman plot.")
