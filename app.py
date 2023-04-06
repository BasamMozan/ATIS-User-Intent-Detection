from flask import Flask, request, render_template

import numpy as np # for numerical computation
import pandas as pd # for data analysis and manipulation

### For data downloading, loading, transformation and DL model download, creating and training
import torch
import torch.nn as NN
from torchvision import transforms, models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

# Initialize Flask application
app = Flask(__name__)

# reading dataset
def read_data(path):
    df = pd.read_csv(path, names=['target', 'data'])
    class_labels = df['target']
    data = df['data']

    return [data, class_labels]



train_data_path = 'archive/atis_intents_train.csv'
test_data_path = 'archive/atis_intents_test.csv'

test_dataset = read_data(test_data_path)
train_dataset = read_data(train_data_path)



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# setting device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# encoding class values
def class_encoder(labels):
    labels_ = set(labels)
    encoder = dict()
    reverse_encoder = dict()

    for i, label in enumerate(labels_):
        encoder[label] = i
        reverse_encoder[i] = label

    return encoder, reverse_encoder


encoder, reverse_encoder = class_encoder(train_dataset[1])

class CustomDataset(Dataset):
    def __init__(self, data, encoder, tokenizer, max_seq):
        self.data = data[0] # sentences
        self.classes = data[1] # labels
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.max_seq = max_seq


    def __len__(self):
        return len(self.classes)

    def __getitem__(self, index):
        text = self.data[index]

        #tokenize the text using tokenizer
        tokens = self.tokenizer.encode_plus(text, max_length=self.max_seq, padding='max_length', truncation=True, return_tensors='pt')

        # get the input ids and convert them to numpy array
        input_ids = tokens['input_ids'][0].numpy()

        # converting label to tensor
        label = torch.tensor(self.encoder[self.classes[index]], dtype=torch.long)

        return input_ids, label

class IntentClassifier(NN.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(IntentClassifier, self).__init__()
        self.fc1 = NN.Linear(input_dim, hidden_dim)
        self.fc2 = NN.Linear(hidden_dim, hidden_dim)
        self.fc3 = NN.Linear(hidden_dim, hidden_dim)
        self.fc4 = NN.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = self.fc1(x)
        x = NN.functional.relu(x)
        # x shape: (batch_size, hidden_dim)
        x = self.fc2(x)
        x = NN.functional.relu(x)
        x = self.fc3(x)
        x = NN.functional.relu(x)
        # x shape: (batch_size, hidden_dim)
        x = self.fc4(x)
        # x shape: (batch_size, output_dim)
        return x



# Load PyTorch model
state_dict = torch.load('intentClassifier.pth')
model = IntentClassifier(300, 128, 8)
model.load_state_dict(state_dict)
model.eval()

# Define prediction function
def predict(query):
    data = [[query], ['atis_flight']]

    data = CustomDataset(data, encoder, tokenizer, 300)
    data = DataLoader(data)

    # model.eval()
    # Loop over the batches in the test data
    with torch.no_grad():
        for (inputs, labels) in data:
            # Forward pass
            inputs = torch.tensor(inputs).float().to(device)
            output = model(inputs)

            # Compute the predicted class and append it to the predictions list
            _, predicted = torch.max(output.data, 1)


    return reverse_encoder[int(predicted.cpu())]

# Define home route
@app.route('/')
def home():
    return render_template('index.html')

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict_route():
    # Get input query from user
    query = request.form['query']
    
    # Perform prediction using pre-trained model
    output = predict(query)
    
    # Display output to user
    return render_template('output.html', output=output)

# Run Flask application
if __name__ == '__main__':
    app.run(debug=True)
