from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Replace with the actual path to your METADATA.csv
metadata_df = pd.read_csv("METADATA_v6.csv")
unique_labels = metadata_df['label'].unique()

le = LabelEncoder()
le.fit(unique_labels)

print("Order of classes in LabelEncoder:", le.classes_)
