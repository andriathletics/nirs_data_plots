import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# Define password
PASSWORD = "nirs_unibern"

# Password lock mechanism
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    password = st.text_input("Enter password", type="password")
    if password == PASSWORD:
        st.session_state.authenticated = True
        st.experimental_rerun()
    else:
        st.error("Incorrect password")
else:
    # Define interventions and variables
    interventions = ['post', 'pre']
    variables = ['SmO2', 'THb', 'TSI']

    # Load data from Excel files
    data = {}
    for intervention in interventions:
        for variable in variables:
            filename = f'masterTable_{intervention}_{variable}.xlsx'
            df = pd.read_excel(filename)
            if intervention in ['pre', 'post']:
                df = df.iloc[:600]  # Limit to 600 seconds for pre and post interventions
            data[(intervention, variable)] = df

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

    # Load the analysis master table
    analysis_master_table = pd.read_excel('analysis_master table.xlsx', sheet_name='pre_post')

    # Selection for correlation plot
    st.sidebar.write("Correlation Plot Selection")
    x_axis = st.sidebar.selectbox('Select X-axis', analysis_master_table.columns[1:])  # Exclude participant ID column
    y_axis = st.sidebar.selectbox('Select Y-axis', analysis_master_table.columns[1:])  # Exclude participant ID column

    # Selection for categories
    st.sidebar.write("Category Selection")
    sex_selected = st.sidebar.multiselect('Select Sex', analysis_master_table['Sex'].unique(), default=analysis_master_table['Sex'].unique())
    moxy_selected = st.sidebar.multiselect('Select Moxy Placement', analysis_master_table['Moxy placement'].unique(), default=analysis_master_table['Moxy placement'].unique())
    fitness_selected = st.sidebar.multiselect('Select Fitness', analysis_master_table['Fitness'].unique(), default=analysis_master_table['Fitness'].unique())
    skin_type_selected = st.sidebar.multiselect('Select Fitzpatrick Skintype', analysis_master_table['Fitzpatrick Skintype'].unique(), default=analysis_master_table['Fitzpatrick Skintype'].unique())

    # Filter data based on selections
    filtered_data = analysis_master_table[
        (analysis_master_table['Sex'].isin(sex_selected)) &
        (analysis_master_table['Moxy placement'].isin(moxy_selected)) &
        (analysis_master_table['Fitness'].isin(fitness_selected)) &
        (analysis_master_table['Fitzpatrick Skintype'].isin(skin_type_selected))
    ]

    # Convert selected columns to numeric, coercing errors
    filtered_data[x_axis] = pd.to_numeric(filtered_data[x_axis], errors='coerce')
    filtered_data[y_axis] = pd.to_numeric(filtered_data[y_axis], errors='coerce')

    # Drop rows with NaN values in the selected columns
    filtered_data = filtered_data.dropna(subset=[x_axis, y_axis])

    # Plot correlation
    fig, ax = plt.subplots()
    sns.regplot(x=filtered_data[x_axis], y=filtered_data[y_axis], ax=ax)
    r_value, p_value = stats.pearsonr(filtered_data[x_axis], filtered_data[y_axis])
    ax.set_title(f'Correlation between {x_axis} and {y_axis}\nR²={r_value**2:.2f}, p={p_value:.2e}')

    # Calculate 95% CI for the correlation coefficient
    n = len(filtered_data)
    se = 1.0 / (n - 3)**0.5
    z = 0.5 * np.log((1 + r_value) / (1 - r_value))
    z_crit = stats.norm.ppf(0.975)
    z_interval = z + np.array([-1, 1]) * z_crit * se
    r_interval = (np.exp(2 * z_interval) - 1) / (np.exp(2 * z_interval) + 1)
    ax.fill_betweenx(ax.get_ylim(), *r_interval, color='gray', alpha=0.2)

    # Display the correlation plot
    st.pyplot(fig)
