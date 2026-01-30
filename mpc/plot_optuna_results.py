import pandas as pd
import plotly.express as px
import os

def main():
    # Define file path
    file_path = "optuna_trials.csv"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Please run train.py first to generate results.")
        return

    # Load data
    df = pd.read_csv(file_path)
    
    # Filter for completed trials only (state == 'COMPLETE')
    if 'state' in df.columns:
        df = df[df['state'] == 'COMPLETE']
    
    # Identify parameter columns (usually start with 'params_')
    param_cols = [col for col in df.columns if col.startswith('params_')]
    
    # Objective value column
    value_col = 'value'
    
    if value_col not in df.columns:
        print(f"Error: '{value_col}' column not found in csv")
        return

    # Create Parallel Coordinates Plot
    print("Generating Parallel Coordinates Plot...")
    
    # Rename columns for better readability in plot
    labels = {col: col.replace('params_', '') for col in param_cols}
    labels[value_col] = 'Loss (Validation)'
    
    fig = px.parallel_coordinates(
        df, 
        dimensions=param_cols + [value_col],
        color=value_col,
        labels=labels,
        color_continuous_scale=px.colors.diverging.Tealrose,
        color_continuous_midpoint=df[value_col].median(),
        title="Optuna Hyperparameter Exploration"
    )

    # Save figure
    output_file = "optuna_space_explored.png"
    fig.write_image(output_file)
    print(f"Figure saved to {output_file}")
    
    # Also show scatter matrix for correlation
    print("Generating Scatter Matrix...")
    fig_scatter = px.scatter_matrix(
        df,
        dimensions=param_cols + [value_col],
        color=value_col,
        labels=labels,
        title="Hyperparameter Scatter Matrix"
    )
    fig_scatter.write_image("optuna_scatter_matrix.png")
    print("Scatter matrix saved to optuna_scatter_matrix.png")

if __name__ == "__main__":
    main()
