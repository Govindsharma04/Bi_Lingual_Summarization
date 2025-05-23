import pandas as pd
from rouge_score import rouge_scorer

# Load the CSV file
df = pd.read_csv('pegasus_hindi_predictions.csv')  # Replace with your file path

# Initialize the ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Function to calculate ROUGE scores
def calculate_rouge_scores(reference, prediction):
    scores = scorer.score(reference, prediction)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

# Calculate ROUGE scores for each row
rouge_scores = df.apply(
    lambda row: calculate_rouge_scores(row['Hindi_summ'], row['generated_hindi_summ']),
    axis=1
)

# Convert the scores into a DataFrame
rouge_df = pd.DataFrame(rouge_scores.tolist())

# Calculate overall average ROUGE scores
average_rouge1 = rouge_df['rouge1'].mean()
average_rouge2 = rouge_df['rouge2'].mean()
average_rougeL = rouge_df['rougeL'].mean()

# Print the overall average ROUGE scores
print(f"Average ROUGE-1 Score: {average_rouge1:.4f}")
print(f"Average ROUGE-2 Score: {average_rouge2:.4f}")
print(f"Average ROUGE-L Score: {average_rougeL:.4f}")
