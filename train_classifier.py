import pickle

data_dict = pickle.load(open('./data.pickle', 'rb'))

# Print the shapes or types of elements to understand the structure
for i, item in enumerate(data_dict['data']):
    if hasattr(item, 'shape'):
        print(f"Item {i}: {item.shape}")
    else:
        print(f"Item {i}: {type(item)}, length: {len(item)}")
import numpy as np

def pad_or_truncate(sequence, length):
    if len(sequence) > length:
        return sequence[:length]
    else:
        return sequence + [0] * (length - len(sequence))

max_length = max(len(item) for item in data_dict['data'])

padded_data = np.array([pad_or_truncate(item, max_length) for item in data_dict['data']])
labels = np.array(data_dict['labels'])
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split data
x_train, x_test, y_train, y_test = train_test_split(padded_data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict and evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
