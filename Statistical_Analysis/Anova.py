import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats import f_oneway


gpt35_responses = pd.read_csv('gpt.csv')
llama2_responses = pd.read_csv('llama2_responses.csv')
human_responses = pd.read_csv('human_responses.csv')

questions_of_interest = [
    "What is the biggest challenge you face as a developer?",
    "When choosing a programming language for a new project, you prioritize:",
    "How do you balance between innovation and meeting project deadlines?"
]

anova_results_adjusted = {}

for question in questions_of_interest:
    combined_responses = pd.concat([
        human_responses[human_responses['Question'].str.contains(question, case=False, na=False)]['Option'],
        gpt35_responses[gpt35_responses['Question'].str.contains(question, case=False, na=False)]['Answer'],
        llama2_responses[llama2_responses['Question'].str.contains(question, case=False, na=False)]['Answer']
    ])
    
    le = LabelEncoder()
    combined_encoded = le.fit_transform(combined_responses)
    
    num_human_responses = human_responses[human_responses['Question'].str.contains(question, case=False, na=False)].shape[0]
    num_gpt35_responses = gpt35_responses[gpt35_responses['Question'].str.contains(question, case=False, na=False)].shape[0]
    
    human_encoded = combined_encoded[:num_human_responses]
    gpt35_encoded = combined_encoded[num_human_responses:num_human_responses + num_gpt35_responses]
    llama2_encoded = combined_encoded[num_human_responses + num_gpt35_responses:]
    
    anova_result = f_oneway(human_encoded, gpt35_encoded, llama2_encoded)
    anova_results_adjusted[question] = anova_result

for question, result in anova_results_adjusted.items():
    print(f"Question: {question}\nF-statistic: {result.statistic}, p-value: {result.pvalue}\n")
