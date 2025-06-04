import pandas as pd
from imblearn.over_sampling import SMOTE

# Load preprocessed data
X_train_scaled = pd.read_csv('X_train_scaled.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Check new class distribution
print(pd.Series(y_train_smote).value_counts())

# Save balanced data
pd.DataFrame(X_train_smote, columns=X_train_scaled.columns).to_csv('X_train_smote.csv', index=False)
pd.Series(y_train_smote).to_csv('y_train_smote.csv', index=False)