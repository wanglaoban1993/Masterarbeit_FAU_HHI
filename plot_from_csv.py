import re
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


current_path = os.getcwd()
print("current path:", current_path)

# files = os.listdir(current_path)
files= ['100_s1_clamp_100.csv', '100_s1_clamp_200.csv', '100_s1_clamp_300.csv', 
        '100_s1_reflect_boundaries_100.csv', '100_s1_reflect_boundaries_200.csv', '100_s1_reflect_boundaries_300.csv',
        '100_s1_reflection_100.csv', '100_s1_reflection_200.csv', '100_s1_reflection_300.csv']

csv_files = {
    '100_s1_clamp_100.csv': 'clamp_reverse_100steps', 
    '100_s1_clamp_200.csv': 'clamp_reverse_200steps', 
    '100_s1_clamp_300.csv': 'clamp_reverse_300steps', 
    '100_s1_reflect_boundaries_100.csv': 'reflect_boundaries_reverse_100steps', 
    '100_s1_reflect_boundaries_200.csv': 'reflect_boundaries_reverse_200steps',
    '100_s1_reflect_boundaries_300.csv': 'reflect_boundaries_reverse_300steps',
    '100_s1_reflection_100.csv': 'reflection_reverse_100steps',
    '100_s1_reflection_200.csv': 'reflection_reverse_200steps',
    '100_s1_reflection_300.csv': 'reflection_reverse_300steps',
}

# Dictionary to store DataFrames
dataframes = {}

# Load each CSV file into a DataFrame and store it in the dictionary
for csv_file, df_name in csv_files.items():
    dataframes[df_name] = pd.read_csv(csv_file)

# Access the DataFrames using their names
clamp_reverse_100steps = dataframes['clamp_reverse_100steps']
clamp_reverse_200steps = dataframes['clamp_reverse_200steps']
clamp_reverse_300steps = dataframes['clamp_reverse_300steps']

reflect_boundaries_reverse_100steps = dataframes['reflect_boundaries_reverse_100steps']
reflect_boundaries_reverse_200steps = dataframes['reflect_boundaries_reverse_200steps']
reflect_boundaries_reverse_300steps = dataframes['reflect_boundaries_reverse_300steps']

reflection_reverse_100steps = dataframes['reflection_reverse_100steps']
reflection_reverse_200steps = dataframes['reflection_reverse_200steps']
reflection_reverse_300steps = dataframes['reflection_reverse_300steps']

# Create a new figure
plt.figure(figsize=(10, 5))
# Plot the first dataset with different transparency (alpha) values

# # Add title and labels
sns.lineplot(x='epoch', y='correct', data= clamp_reverse_100steps, label='reverse_100steps',  alpha=0.9)
sns.lineplot(x='epoch', y='correct', data= clamp_reverse_200steps, label='reverse_200steps',  alpha=0.9)
sns.lineplot(x='epoch', y='correct', data= clamp_reverse_300steps, label='reverse_300steps',  alpha=0.9)
sns.lineplot(x='epoch', y='correct', data= reflect_boundaries_reverse_100steps, label='reverse_100steps',  alpha=0.9)
sns.lineplot(x='epoch', y='correct', data= reflect_boundaries_reverse_200steps, label='reverse_200steps',  alpha=0.9)
sns.lineplot(x='epoch', y='correct', data= reflect_boundaries_reverse_300steps, label='reverse_300steps',  alpha=0.9)
sns.lineplot(x='epoch', y='correct', data= reflection_reverse_100steps, label='reverse_100steps',  alpha=0.9)
sns.lineplot(x='epoch', y='correct', data= reflection_reverse_200steps, label='reverse_200steps',  alpha=0.9)
sns.lineplot(x='epoch', y='correct', data= reflection_reverse_300steps, label='reverse_300steps',  alpha=0.9)

# end_correct_100steps= reverse_100steps['correct'].iloc[-1]
# end_correct_200steps= reverse_200steps['correct'].iloc[-1]
# end_correct_300steps= reverse_300steps['correct'].iloc[-1]
# print(end_correct_100steps, end_correct_200steps, end_correct_300steps)

# max_correct_100steps= reverse_100steps['correct'].max()
# max_correct_200steps= reverse_200steps['correct'].max()
# max_correct_300steps= reverse_300steps['correct'].max()
# print(max_correct_100steps, max_correct_200steps, max_correct_300steps)

# Add title and labels
plt.title(f"Combined plots with clamp, reflectboundaires, reflection in 100. 200. 300steps", fontsize=10)
plt.xlabel('Epoch')
plt.ylabel('Correct')

# Show the legend
plt.legend()
# Save the combined plot
plt.savefig('combined_100_s1_all_100_200_300.png')
# Display the plot
plt.show()
plt.close()