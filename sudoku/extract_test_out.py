import re
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


current_path = os.getcwd()
print("current path:", current_path)

files = os.listdir(current_path)
print("everything unter current path: ", len(files))

def extract_data_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    epochs = re.findall(r'epoch_(\d+)', content)
    accuracies = re.findall(r'Sudoku accuracy: (\d+\.\d+)%', content)
    
    return list(zip(epochs, accuracies))

def save_to_csv(data, output_path):
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'correct']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for epoch, correct in data:
            writer.writerow({'epoch': epoch, 'correct': correct})

# List of input and output file paths
input_files = [
    '1413245_test_100_s1_reflection_100.out',
    '1413246_test_100_s1_reflection_200.out',
    '1414218_test_100_s1_reflection_300.out'

]
output_files = [
    '100_s1_reflection_100.csv',
    '100_s1_reflection_200.csv',
    '100_s1_reflection_300.csv'
]

# Process each file and save results to CSV
for input_file, output_file in zip(input_files, output_files):
    data = extract_data_from_file(input_file)
    save_to_csv(data, output_file)
print("CSV files created successfully.")

# List of CSV files and corresponding DataFrame names
# csv_files = {
#     '400_s1_clamp_400.csv': 'df_original',
#     '400_s1_clamp_800.csv': 'df_reflectboundaries',
#     '400_s1_clamp_1200.csv': 'df_reflection'
# }
csv_files = {
    '100_s1_reflection_100.csv': 'reverse_100steps',
    '100_s1_reflection_200.csv': 'reverse_200steps',
    '100_s1_reflection_300.csv': 'reverse_300steps'
}

# Dictionary to store DataFrames
dataframes = {}

# Load each CSV file into a DataFrame and store it in the dictionary
for csv_file, df_name in csv_files.items():
    dataframes[df_name] = pd.read_csv(csv_file)

# Access the DataFrames using their names
# df_original = dataframes['df_original']
# df_reflectboundaries = dataframes['df_reflectboundaries']
# df_reflection = dataframes['df_reflection']

reverse_100steps = dataframes['reverse_100steps']
reverse_200steps = dataframes['reverse_200steps']
reverse_300steps = dataframes['reverse_300steps']

# Create a new figure
plt.figure(figsize=(10, 5))
# Plot the first dataset with different transparency (alpha) values
# sns.lineplot(x='epoch', y='correct', data=df_original, label='DDSM', color= 'blue', alpha=0.9)
# sns.lineplot(x='epoch', y='correct', data=df_reflectboundaries, label='Reflectboundaries', color= 'red', alpha=0.9)
# sns.lineplot(x='epoch', y='correct', data=df_reflection, label='Reflection', color= 'green', alpha=0.9)

# end_correct_original= df_original['correct'].iloc[-1]
# end_correct_reflectboundaries= data=df_reflectboundaries['correct'].iloc[-1]
# end_correct_reflection= data=df_reflection['correct'].iloc[-1]

# # Add title and labels
# plt.title(f"DDSM End: {end_correct_original}\nReflectboundaries End: {end_correct_reflectboundaries}\nReflection End: {end_correct_reflection}", fontsize=10)
# plt.xlabel('Epoch')
# plt.ylabel('Correct')

sns.lineplot(x='epoch', y='correct', data=reverse_100steps, label='reverse_100steps', color= 'blue', alpha=0.9)
sns.lineplot(x='epoch', y='correct', data=reverse_200steps, label='reverse_200steps', color= 'red', alpha=0.9)
sns.lineplot(x='epoch', y='correct', data=reverse_300steps, label='reverse_300steps', color= 'green', alpha=0.9)

end_correct_100steps= reverse_100steps['correct'].iloc[-1]
end_correct_200steps= reverse_200steps['correct'].iloc[-1]
end_correct_300steps= reverse_300steps['correct'].iloc[-1]
print(end_correct_100steps, end_correct_200steps, end_correct_300steps)

max_correct_100steps= reverse_100steps['correct'].max()
max_correct_200steps= reverse_200steps['correct'].max()
max_correct_300steps= reverse_300steps['correct'].max()
print(max_correct_100steps, max_correct_200steps, max_correct_300steps)

# Add title and labels
plt.title(f"End,Max with 100steps: {end_correct_100steps, max_correct_100steps}\nEnd,Max with 200steps: {end_correct_200steps, max_correct_200steps}\nEnd,Max with 300steps: {end_correct_300steps, max_correct_300steps}", fontsize=10)
plt.xlabel('Epoch')
plt.ylabel('Correct')

# Show the legend
plt.legend()
# Save the combined plot
plt.savefig('combined_100_s1_reflection_100_200_300.png')
# Display the plot
plt.show()
plt.close()