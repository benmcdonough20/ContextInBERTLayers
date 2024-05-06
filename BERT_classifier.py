import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig
import tqdm

import datasets #to be replaced with CFG generator

DEVICE = torch.device('cuda')
CONFIG = BertConfig.from_pretrained('bert-base-uncased', show_hidden_states = True)

sst2 = datasets.load_dataset("sst2")

class BERTHiddenStateClassifier(nn.Module):

  def __init__(self):
    super(BERTHiddenStateClassifier, self).__init__()

    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # from https://huggingface.co/docs/transformers/model_doc/bert#tfbertmodel
    # now "bert(**input)" will contain a key for all hidden layers (13)
    self.bert = BertModel.from_pretrained("bert-base-uncased", config = CONFIG).to(device=DEVICE)
    self.output = nn.Sequential(
      nn.Linear(
        in_features = 768, #represents a BERT encoding of a single word. This should be enough but
                            #if we need more, we can just concatenate them together
        out_features = 1 #number of categories to classify over, e.g. 
                          #corresponding to the # of nouns in the grammar if we do that
      ),
      nn.Sigmoid()
    )

  # vector, it should return the output
  def forward(self, inp):

    #encode and move to GPU
    encoded_input = self.tokenizer(inp, return_tensors="pt")
    for key in encoded_input:
      encoded_input[key] = encoded_input[key].to(device=DEVICE)

    # dict list contains 3 dimensional array by 1) sentence 2) word 3) entries in hidden state.
    # Word [0] is always CLS. Not sure why this token is special
    emb = self.bert(**encoded_input)["last_hidden_state"][0][0]
    emb = emb.detach() #I guess this moves it out of GPU mem?
    outp = self.output(emb)

    return outp

  def predict(self, sentence): #classify and generate prediction to compute loss
    prob = self.forward(sentence)
    return (prob > .5, prob)


loss_function = nn.BCELoss()
def compute_validation_loss(model):

  total_loss = 0
  total_correct = 0
  count_examples = len(sst2["validation"])

  for example in tqdm.tqdm(sst2["validation"]):

    sentence, correct_label = example["sentence"], example["label"]

    label = torch.FloatTensor([correct_label]).to(device=DEVICE)
    predicted = model(sentence)
    total_loss += loss_function(predicted, label)
    total_correct += int((predicted.item() > 0.5) == example["label"])

  return total_loss / count_examples, total_correct / count_examples

cls = BERTHiddenStateClassifier().to(device=DEVICE)
"""
optimizer = torch.optim.Adam(cls.parameters(), lr=0.005)

loss = 0
for index, example in enumerate(sst2["train"]):

  # Get the sentence and label for this example
  sentence = example["sentence"]
  label = torch.FloatTensor([example["label"]]).to(device=DEVICE)

  # Get the model's prediction
  predicted = cls(sentence)
  # Compute the loss
  loss += loss_function(predicted, label)

  if index % 64 == 0: #Can we improve this naive batching strategy?
    loss.backward() #loss is a tensor and contains a backpropagation method
    optimizer.step() #step and then zero out
    optimizer.zero_grad()
    loss = 0

  #monitor performance
  if index % 1000 == 0:
    loss, accuracy = compute_validation_loss(cls)
    print(index, loss.item(), accuracy)

#functions defined in the classifer class can now be used
cls.predict("I hate this movie!")
"""
example = sst2["train"][0]
sentence = example["sentence"]
label = torch.FloatTensor([example["label"]]).to(device=DEVICE)

# Get the model's prediction
predicted = cls(sentence)
# Compute the loss

loss, accuracy = compute_validation_loss(cls)
print(index, loss.item(), accuracy)