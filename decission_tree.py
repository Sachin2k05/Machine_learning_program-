import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
# Step 1: Dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny',
                'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild',
                    'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High',
                 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak',
             'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No',
             'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}
# Step 2: Load dataset
df = pd.DataFrame(data)
# Step 3: Encode all columns separately
encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # Save encoder for later decoding
# Step 4: Split features and target
X = df.drop(columns='Play')
y = df['Play']
# Step 5: Train decision tree with ID3 (entropy)
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)
# Step 6: Predict a new sample (e.g., Sunny, Cool, High, Strong)
# Use saved encoders to encode new data
sample = [[
    encoders['Outlook'].transform(['Sunny'])[0],
    encoders['Temperature'].transform(['Cool'])[0],
    encoders['Humidity'].transform(['High'])[0],
    encoders['Wind'].transform(['Strong'])[0]
]]
# Step 7: Predict and decode
prediction = model.predict(sample)
predicted_label = encoders['Play'].inverse_transform(prediction)
# Step 8: Output
print("Prediction for new sample (Sunny, Cool, High, Strong):", predicted_label[0])
