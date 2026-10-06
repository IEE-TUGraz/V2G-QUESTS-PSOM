import os
import pandas as pd
import plotly.graph_objects as go

# This script processes multiple Excel files containing energy consumption data and puts it into one DataFrame which is then saved as a CSV file.
# It also generates a time series plot of the energy consumption data.


# Directory where the files are stored
directory = os.path.join(os.getcwd(), 'raw_data', '3810 - (xxx)')

# Dictionary to store each file's data
data_frames = {}

# Loop through all Excel files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".xlsx"):
        print(f"Processing file: {filename}")
        file_id = os.path.splitext(filename)[0]  # Use the filename without extension
        # split file_id at '_' if it exists
        if '_' in file_id:
            file_id = file_id.split('_')[0]
        file_id = "3810-" + file_id  # Add prefix to file_id
        file_path = os.path.join(directory, filename)
        
        # Load file
        df = pd.read_excel(file_path)
        
        # Parse time and extract relevant columns
        df['Data/Hora'] = pd.to_datetime(df['Data/Hora'])
        df = df[['Data/Hora', 'Energia ativa (kWh)']].rename(columns={'Energia ativa (kWh)': file_id, 'Data/Hora': 'Time'})
        df.set_index('Time', inplace=True)
        
        data_frames[file_id] = df


# Merge all DataFrames on time index
merged_df = pd.concat(data_frames.values(), axis=1)
# save to csv
merged_df.to_csv(os.path.join('intermediate_data', 'consumption_data_new_CP7.csv'), index=True)
# Display a preview
print(merged_df.tail(10))


def plot_energy_timeseries(df):
    fig = go.Figure()

    for column in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[column],
            mode='lines',
            name=column
        ))

    fig.update_layout(
        title="Energy Usage Over Time",
        xaxis_title="Time",
        yaxis_title="Energy (kWh)",
        hovermode="x unified"
    )

    fig.show()

plot_energy_timeseries(merged_df)