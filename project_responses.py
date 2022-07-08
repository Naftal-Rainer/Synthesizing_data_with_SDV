from operator import index
# Import the required librares

import pandas as pd 
from sdv.tabular import GaussianCopula
from sdv.evaluation import evaluate

# Load the data
data = pd.read_excel('Project responses.xlsx')

# Peek at the loaded data
print(data.head())

#  Initialize the single table SDV model
model = GaussianCopula()

# For training, initiate the class and fit the data
model.fit(data)

# The `sample` attribute from the model, we obtain the randomized synthetic data. How much data you want depends on the number you pass into the `sample` attribute.
sample = model.sample(500)

# Save the output
sample.to_excel('Synthetic.xlsx', index=False)
print(sample.head())

# For evaluation, compare the real dataset with the sample dataset and evaluate using the many tests/metrics available.
# I focused on the Kolmogorov-Smirnov (KS) and Chi-Squared (CS) tests.
score = evaluate(sample, data, metrics=['CSTest', 'KSTest'], aggregate=False)
print(score)