import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import pandas as pd

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    print(f"Loading data from {data_dir}...")
    
    try:
        x_train = torch.load(os.path.join(data_dir, 'x_train.pt'))
        u_train = torch.load(os.path.join(data_dir, 'u_train.pt'))
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # Convert to numpy
    X = x_train.cpu().numpy()
    U = u_train.cpu().numpy()
    
    print(f"X shape: {X.shape}")
    print(f"U shape: {U.shape}")

    # Infer dimensions
    nu = U.shape[1]
    # Assuming standard structure: [x0 (nu), xd (nu), q (nu), lower (nu), upper (nu)]
    # But usually x contains at least [x0, xd].
    # Let's verify width. 
    # If X shape is larger than 2*nu, we assume extra columns are q/constraints.
    
    n_states_to_plot = 2 * nu # Plot both x0 and xd
    
    if X.shape[1] < n_states_to_plot:
        print(f"Warning: X width ({X.shape[1]}) is smaller than expected 2*nu ({2*nu}). Plotting all columns.")
        n_states_to_plot = X.shape[1]

    # Create a DataFrame for easier plotting with Seaborn
    # Naming columns
    col_names = []
    for i in range(nu):
        col_names.append(f"x0_{i}")
    for i in range(nu, n_states_to_plot):
        col_names.append(f"xd_{i-nu}")
        
    df_states = pd.DataFrame(X[:, :n_states_to_plot], columns=col_names[:n_states_to_plot])
    
    # 1. Plot Histograms of States
    print("Generating State Histograms...")
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(df_states.columns):
        plt.subplot(int(np.ceil(len(df_states.columns)/4)), 4, i+1)
        sns.histplot(df_states[col], kde=True, bins=50)
        plt.title(f"Distribution of {col}")
        plt.xlabel("Value")
        plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "mpc", "state_histograms.png"))
    print("Saved state_histograms.png")
    
    # 2. Plot Pairplot (Scatter Matrix) of States
    # Downsample for speed if dataset is huge
    N_samples = 1000000000
    if len(df_states) > N_samples:
        print(f"Downsampling to {N_samples} points for scatter matrix...")
        df_sample = df_states.sample(N_samples)
    else:
        df_sample = df_states

    print("Generating State Scatter Matrix (this may take a moment)...")
    pp = sns.pairplot(df_sample, diag_kind="kde", plot_kws={'alpha': 0.5, 's': 10})
    pp.fig.suptitle("State Space Exploration (Pairplot)", y=1.02)
    pp.savefig(os.path.join(base_dir, "mpc", "state_pairplot.png"))
    print("Saved state_pairplot.png")

    # 3. Plot Inputs Distributions
    print("Generating Input Histograms...")
    df_inputs = pd.DataFrame(U, columns=[f"u_{i}" for i in range(nu)])
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(df_inputs.columns):
        plt.subplot(1, nu, i+1)
        sns.histplot(df_inputs[col], kde=True, bins=50, color='orange')
        plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "mpc", "input_histograms.png"))
    print("Saved input_histograms.png")

    # 4. Temporal view (optional, if data is sequential it might look like a mess or traces)
    # Since data is shuffled in process_data_shuff, temporal plotting is not useful here.

if __name__ == "__main__":
    main()
