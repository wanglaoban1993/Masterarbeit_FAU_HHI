
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.stats import beta, gaussian_kde, wasserstein_distance
from scipy.integrate import simps
import argparse

def parse_args():
    parser = argparse.ArgumentParser("Pre generate jacobi process values with specified number of "
                                     "categories and time points")

    parser.add_argument("--out_path", type=str,
                        help="Path to output directory, where precomputed noise will be saved",
                        )
    parser.add_argument("--file_path", type=str,
                    help="Path to output directory, where precomputed noise will be imported",
                    )
    return parser.parse_args()
args= parse_args()

# Ensure the directory exists
#output_dir = "path_simulation_plots_std400/sab/reflection"
output_dir = args.out_path
os.makedirs(output_dir, exist_ok=True)

def calculate_kl_divergence(p, q, x):
    """Calculate KL divergence between two distributions P and Q over x."""
    mask = (q > 0) & (p > 0)  # Avoid division by zero and log of zero
    p = p[mask]
    q = q[mask]
    x = x[mask]
    ratio = p / q
    log_ratio = np.log(ratio)
    kl_div = simps(p * log_ratio, x)
    return kl_div

def average_density_plot(densities, x_values, plot_name):
    """Plot average density distribution over multiple KDEs."""
    avg_density = np.mean(densities, axis=0)
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, avg_density, label='Average Density')
    plt.xlabel('X')
    plt.ylabel('Density')
    plt.title('Average Density Distribution')
    plt.legend()
    plt.grid(True)
    file_path = os.path.join(output_dir, f'{plot_name}.png')
    plt.savefig(file_path)
    plt.close()

def load_and_process_data(file_path, num_elements=5000):
    """Load data from .pth file and process it."""
    loaded_object = torch.load(file_path)
    v_one = loaded_object[0]
    v_zero = loaded_object[1]
    #timepoints = loaded_object[4]
    timepoints = loaded_object[2]

    num_samples = min(v_one.shape[0], 20000)
    num_channels = v_one.shape[2]
    indices_v_zero = torch.randperm(v_zero.shape[0])[:num_elements]
    indices_v_one = torch.randperm(v_one.shape[0])[:num_elements]
    selected_v_zero = v_zero[indices_v_zero]
    selected_v_one = v_one[indices_v_one]
    merged_tensor = torch.cat((selected_v_zero, selected_v_one), dim=0)
    return merged_tensor, timepoints, num_samples, num_channels

def compute_kl_and_wasserstein(merged_samples, num_channels, beta_values, alpha=1, num_iterations=20):
    """
    Compute KL divergence and Wasserstein distance for each channel.
    
    Parameters:
    - merged_samples: List of tensors containing samples for each channel.
    - num_channels: Number of channels.
    - beta_values: List of beta values for the Beta distributions.
    - alpha: Alpha value for the Beta distributions.
    - num_iterations: Number of iterations for subsampling.
    
    Returns:
    - kl_results: List of (mean, std) for KL divergences.
    - wasserstein_results: List of (mean, std) for Wasserstein distances.
    - kl_divergences_list: List of KL divergences for each channel across iterations.
    - wasserstein_distances_list: List of Wasserstein distances for each channel across iterations.
    """
    kl_results = []
    wasserstein_results = []
    kl_divergences_list = []
    wasserstein_distances_list = []
    x = np.linspace(0, 1, 1000)
    
    for channel in range(num_channels):
        kl_divergences = []
        wasserstein_distances = []
        densities = []
        
        for _ in range(num_iterations):
            # Select 5000 samples from each group
            samples_zero = merged_samples[channel][:, -1][torch.randperm(merged_samples[channel].shape[0])[:5000]].numpy()
            samples_one = merged_samples[channel][:, -1][torch.randperm(merged_samples[channel].shape[0])[:5000]].numpy()
            # samples_zero = merged_samples[channel][:, -1][torch.randperm(merged_samples[channel].shape[0])[:4000]].numpy()
            # samples_one = merged_samples[channel][:, -1][torch.randperm(merged_samples[channel].shape[0])[:4000]].numpy()
            combined_samples = np.concatenate((samples_zero, samples_one))
            
            # KDE for the current iteration
            kde = gaussian_kde(combined_samples, bw_method=0.1)
            p = kde(x)
            densities.append(p)
            
            # Beta distribution PDF
            q = beta.pdf(x, alpha, beta_values[channel])
            
            # Compute KL divergence
            kl_div = calculate_kl_divergence(p, q, x)
            kl_divergences.append(kl_div)
            
            # Compute Wasserstein distance
            wasser_dist = wasserstein_distance(combined_samples, beta.rvs(alpha, beta_values[channel], size=combined_samples.shape[0]))
            wasserstein_distances.append(wasser_dist)
        
        # Compute mean and std for KL and Wasserstein
        kl_mean = np.mean(kl_divergences)
        kl_std = np.std(kl_divergences)
        wasserstein_mean = np.mean(wasserstein_distances)
        wasserstein_std = np.std(wasserstein_distances)
        
        kl_results.append((kl_mean, kl_std))
        wasserstein_results.append((wasserstein_mean, wasserstein_std))
        kl_divergences_list.append(kl_divergences)
        wasserstein_distances_list.append(wasserstein_distances)
        
        # Plot average density distribution
        average_density_plot(densities, x, f"Channel_{channel + 1}_Average_Density")
    
    return kl_results, wasserstein_results, kl_divergences_list, wasserstein_distances_list

def visualize_diffusion_trajectories(timepoints, samples_to_plot, num_samples, plot_name, title="Diffusion Process"):
    """
    Visualize the diffusion process by plotting trajectories of multiple samples.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(num_samples):
        ax.plot(timepoints, samples_to_plot[i, :], alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("X")
    file_path = os.path.join(output_dir, f"{plot_name}.png")
    fig.savefig(file_path)
    plt.close(fig)
    return file_path

def plot_kde_with_beta(merged_samples, num_channels, beta_values, kl_divergences, wasserstein_distances, alpha=1):
    """
    Plot KDE and Beta distributions for each channel with KL divergence and Wasserstein distance annotations.
    
    Parameters:
    - merged_samples: List of tensors containing samples for each channel.
    - num_channels: Number of channels.
    - beta_values: List of beta values for the Beta distributions.
    - kl_divergences: List of KL divergences for each channel.
    - wasserstein_distances: List of Wasserstein distances for each channel.
    - alpha: Alpha value for the Beta distributions.
    """
    colors = ['blue', 'orange', 'red', 'pink', 'yellow', 'purple', 'green', 'cyan']
    x = np.linspace(0, 1, 1000)
    
    for channel in range(num_channels):
        plt.figure(figsize=(8, 6))
        sns.kdeplot(merged_samples[channel][:, -1], fill=True, bw_method=0.1, label=f'Channel {channel + 1} KDE', color='blue')
        
        y = beta.pdf(x, alpha, beta_values[channel])
        plt.plot(x, y, label=f'Beta({alpha},{beta_values[channel]})', color='red', linestyle='--')
        
        plt.title(f'KDE Plot and Beta Distribution for Channel {channel + 1}')
        plt.xlabel('X')
        plt.ylabel('Density')
        plt.legend()
        
        # Add KL divergence and Wasserstein distance annotations
        plt.text(0.5, 0.9, f'KL Divergence: {kl_divergences[channel]:.4f}', horizontalalignment='center',
                 verticalalignment='center', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        plt.text(0.5, 0.85, f'Wasserstein Distance: {wasserstein_distances[channel]:.4f}', horizontalalignment='center',
                 verticalalignment='center', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        file_path = os.path.join(output_dir, f'Channel_{channel + 1}_KDE_Beta.png')
        plt.savefig(file_path)
        plt.close()

def plot_combined_kde(merged_samples, num_channels):
    """Plot combined KDE for the final timepoint in each channel."""
    plt.figure(figsize=(16, 9))
    colors = ['blue', 'orange', 'red', 'pink', 'yellow', 'purple', 'green', 'cyan']
    for i in range(num_channels):
        sns.kdeplot(merged_samples[i][:,-1], fill=True, bw_method=0.1, color=colors[i], label=f'Channel {i + 1}')
    U = torch.rand(merged_samples[0].shape[0])
    sns.kdeplot(U, fill=True, bw_method=0.1, color='gray', label='Uniform')
    plt.legend()
    plt.title('KDE Plots of Final Timepoint for Each Channel')
    plt.xlabel('X')
    plt.ylabel('Density')
    file_path = os.path.join(output_dir, 'distribution_density_merge_plot.png')
    plt.savefig(file_path)
    plt.show()

def plot_combined_metrics(kl_results, wasserstein_results):
    """Plot the mean and standard deviation of KL divergence and Wasserstein distance for each channel on the same plot."""
    num_channels = len(kl_results)
    channels = np.arange(num_channels) + 1

    # Extract means and standard deviations
    kl_means = [result[0] for result in kl_results]
    kl_stds = [result[1] for result in kl_results]
    wasserstein_means = [result[0] for result in wasserstein_results]
    wasserstein_stds = [result[1] for result in wasserstein_results]

    plt.figure(figsize=(14, 7))

    # Plot KL divergence
    plt.errorbar(channels, kl_means, yerr=kl_stds, fmt='-o', label='KL Divergence', capsize=5, color='blue')

    # Plot Wasserstein distance
    plt.errorbar(channels, wasserstein_means, yerr=wasserstein_stds, fmt='-o', label='Wasserstein Distance', capsize=5, color='red')

    plt.xlabel('Channel')
    plt.ylabel('Value')
    plt.title('KL Divergence and Wasserstein Distance Mean and Standard Deviation for Each Channel')
    plt.xticks(channels, [f'Channel {i}' for i in channels])
    plt.legend()
    plt.grid(True)
    file_path = os.path.join(output_dir, 'Combined_Metrics_Summary.png')
    plt.savefig(file_path)
    plt.close()

# Example usage
#file_path = './sudoku/steps400.cat9.time1.0.samples50000.reflection.sab.pth'
file_path = args.file_path
merged_tensor, timepoints, num_samples, num_channels = load_and_process_data(file_path)

beta_values = [8, 7, 6, 5, 4, 3, 2, 1]
alpha = 1

# Prepare merged samples for KDE and further analysis
merged_samples = []
for channel in range(num_channels):
    samples = merged_tensor[:, :, channel].squeeze()
    merged_samples.append(samples)

# Compute KL divergences and Wasserstein distances
kl_results, wasserstein_results, kl_divergences_list, wasserstein_distances_list = compute_kl_and_wasserstein(merged_samples, num_channels, beta_values, alpha)

print("KL divergences for each channel (mean, std):", kl_results)
#print("kl_divergences_list", kl_divergences_list)
print()
print("Wasserstein distances for each channel (mean, std):", wasserstein_results)
#print("wasserstein_distances_list", wasserstein_distances_list)
print()
import csv
# Column names for the CSV files
column_names = ["Mean", "Std"]

# Specify the output directory
#output_dir = "path_simulation_plots_std100/reflection"

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define file paths for the CSV files
kl_divergences_path = os.path.join(output_dir, "kl_divergences.csv")
wasserstein_distances_path = os.path.join(output_dir, "wasserstein_distances.csv")

kl_divergences_list_path= os.path.join(output_dir, "kl_divergences_list.csv")
wasserstein_distances_list_path= os.path.join(output_dir, "wasserstein_distances_list.csv")

# Save KL divergences to a CSV file
with open(kl_divergences_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(column_names)
    writer.writerows(kl_results)

# Save Wasserstein distances to a CSV file
with open(wasserstein_distances_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(column_names)
    writer.writerows(wasserstein_results)
print(f"Data saved to {kl_divergences_path} and {wasserstein_distances_path}")

import pandas as pd
data = []
for channel_idx in range(1, 9):  # Channel indices from 1 to 8
    data.append([channel_idx] + kl_divergences_list[channel_idx - 1])  # Add channel index and KL values

#df = pd.DataFrame(data, columns=["Channel"] + [f"Iteration {i+1}" for i in range(10)])
df = pd.DataFrame(data, columns=["Channel"] + [f"Iteration {i+1}" for i in range(20)])
# Save the DataFrame to a CSV file
df.to_csv(kl_divergences_list_path, index=False)
print("KL divergences saved")

data = []
for channel_idx in range(1, 9):  # Channel indices from 1 to 8
    data.append([channel_idx] + wasserstein_distances_list[channel_idx - 1])  # Add channel index and KL values

#df = pd.DataFrame(data, columns=["Channel"] + [f"Iteration {i+1}" for i in range(10)])
df = pd.DataFrame(data, columns=["Channel"] + [f"Iteration {i+1}" for i in range(20)])
# Save the DataFrame to a CSV file
df.to_csv(wasserstein_distances_list_path, index=False)
print("wasserstein_distances saved")


# Plot KDE with Beta distributions and annotate with KL and Wasserstein values
plot_kde_with_beta(merged_samples, num_channels, beta_values, [kl[0] for kl in kl_results], [wd[0] for wd in wasserstein_results], alpha)

# Plot combined KDE for comparison
plot_combined_kde(merged_samples, num_channels)

# Plot combined metrics for KL divergence and Wasserstein distance
plot_combined_metrics(kl_results, wasserstein_results)
