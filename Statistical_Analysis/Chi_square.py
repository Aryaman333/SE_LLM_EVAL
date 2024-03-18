import pandas as pd
from scipy.stats import chi2_contingency
from utils import option_mapping, contingency_table
import numpy as np

gpt3_5_data = pd.read_csv('gpt.csv')
llama2_data = pd.read_csv('llama2_responses.csv')
human_data = pd.read_csv('human_responses.csv')

human_data['Option ID'] = human_data['Option'].map(option_mapping)

gpt3_5_option_counts = gpt3_5_data['Option IDs'].str.get_dummies(sep=', ').sum()
llama2_option_counts = llama2_data['Answer'].str.get_dummies(sep=', ').sum()
human_option_counts = human_data['Option ID'].value_counts()

option_counts_comparison = pd.DataFrame({
    'ChatGPT 3.5': gpt3_5_option_counts,
    'LLaMA 2': llama2_option_counts,
    'Human': human_option_counts
}).fillna(0) 

chi2, p, dof, expected = chi2_contingency(option_counts_comparison.iloc[:, :2].T)

print(f"Chi-Square Statistic: {chi2}")
print(f"p-value: {p}")
print(f"Degrees of Freedom: {dof}")
print("Expected Frequencies if no association between model type and option selection:")
print(expected)

n = contingency_table.sum()

k = min(contingency_table.shape) 


cramers_v = np.sqrt(chi2 / (n * (k - 1)))

print(f"Cramér's V: {cramers_v}")
