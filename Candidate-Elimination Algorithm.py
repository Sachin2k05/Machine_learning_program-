import pandas as pd

# Read the CSV file
data = pd.read_csv('data.csv')

# Separate features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Initialize S and G
num_attributes = X.shape[1]
S = ['ϕ'] * num_attributes
G = [['?'] * num_attributes]

# Candidate-Elimination Algorithm
for i in range(len(X)):
    if y[i] == 'Yes':
        # Update S for positive example
        for j in range(num_attributes):
            if S[j] == 'ϕ':
                S[j] = X[i][j]
            elif S[j] != X[i][j]:
                S[j] = '?'
        # Remove G hypotheses that don't match
        G = [g for g in G if all(g[j] == '?' or g[j] == X[i][j] for j in range(num_attributes))]
    else:
        # For negative example, specialize G
        new_G = []
        for g in G:
            for j in range(num_attributes):
                if g[j] == '?':
                    for value in data.iloc[:, j].unique():
                        if value != X[i][j]:
                            new_hyp = g.copy()
                            new_hyp[j] = value
                            if all(S[k] == '?' or new_hyp[k] == '?' or new_hyp[k] == S[k] for k in range(num_attributes)):
                                new_G.append(new_hyp)
        G = new_G

# Print final hypothesis
print("Final Specific Hypothesis (S):", S)
print("\nFinal General Hypotheses (G):")
for g in G:
    print(g)
