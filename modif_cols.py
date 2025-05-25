import pandas as pd
import numpy as np

def tidy_emg_imu_as_measured(df):
    # Identify columns to melt (all sensor columns)
    measurement_cols = [c for c in df.columns if any(
        sensor in c for sensor in ['RDelt', 'LDelt', 'RBicep', 'LBicep'])]
    id_vars = [c for c in df.columns if c not in measurement_cols]
    # Melt
    df_long = df.melt(id_vars=id_vars, value_vars=measurement_cols,
                      var_name='Measurement', value_name='Value')
    # Extract BodyPart and Signal
    df_long['BodyPart'] = df_long['Measurement'].str.extract(r'^(RDelt|LDelt|RBicep|LBicep)')
    df_long['Signal'] = df_long['Measurement'].str.replace(r'^(RDelt|LDelt|RBicep|LBicep)_', '', regex=True)
    
    # Pivot so each signal is a separate column
    df_wide = df_long.pivot_table(
        index=id_vars + ['BodyPart'],
        columns='Signal',
        values='Value'
    ).reset_index()
    # flatten columns if needed
    df_wide.columns.name = None
    df_wide.columns = [str(col) for col in df_wide.columns]  
    return df_wide