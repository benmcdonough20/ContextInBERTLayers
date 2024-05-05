import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig

DEVICE = torch.device('cuda')
CONFIG = BertConfig.from_pretrained('bert-base-uncased', show_hidden_states = True)

import json

train_file = open("/home/ben/Documents/Repos/ContextInBERTLayers/train_data.json", "r")
train_data = json.load(train_file)

classes = list(set([t["class"] for t in train_data]))

def dirac_mass(cat):
  ret = [0]*len(classes)
  ret[classes.index(cat)] = 1
  return ret

class BERTHiddenStateClassifier(nn.Module):

  def __init__(self):
    super(BERTHiddenStateClassifier, self).__init__()

    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # from https://huggingface.co/docs/transformers/model_doc/bert#tfbertmodel
    # now "bert(**input)" will contain a key for all hidden layers (13)
    self.bert = BertModel.from_pretrained("bert-base-uncased", config = CONFIG).to(device=DEVICE)

    """
    replace this with RNN! 
    """
    self.output = nn.Sequential(
      nn.Linear(
        in_features = 768, #represents a BERT encoding of a single word. This should be enough but
                            #if we need more, we can just concatenate them together
        out_features = len(classes) #number of categories to classify over, e.g. 
                          #corresponding to the # of nouns in the grammar if we do that
      ),
      nn.Softmax()
    )

  # vector, it should return the output
  def forward(self, inp):

    #encode and move to GPU
    tokens_idxs = inp[0]
    encoded_input = self.tokenizer(inp[1], return_tensors = "pt")
    for key, value in encoded_input.items():
      encoded_input[key] = value.to(device = DEVICE)

    """
    For now, just using the first of the tokens to classify. Replace with RNN
    """
    emb = self.bert(**encoded_input)["last_hidden_state"][0][tokens_idxs[0]]
    
    emb = emb.detach() #I guess this moves it out of GPU mem?
    outp = self.output(emb)

    return outp

  def predict(self, noun_tokens, sentence): #classify and generate prediction to compute loss
    prob = self.forward(sentence)
    return classes[torch.topk(prob, 1)[1].item()]

loss_function = nn.CrossEntropyLoss()
def compute_validation_loss(model):

  total_loss = 0
  total_correct = 0
  count_examples = len(train_data)

  for example in train_data:

    tokenized_sentence, token_idxs, correct_label = example['input'], example['noun_tok_poses'], example['class']

    ideal_dist = torch.Tensor(dirac_mass(correct_label)).to(device=DEVICE)
    predicted = model([token_idxs, tokenized_sentence])
    total_loss += loss_function(predicted, ideal_dist)
    total_correct += int(classes[torch.topk(predicted, 1)[1].item()] == correct_label)

  return total_loss / count_examples, total_correct / count_examples


cls = BERTHiddenStateClassifier().to(device=DEVICE)
optimizer = torch.optim.Adam(cls.parameters(), lr=0.005)

loss = 0
for index, example in enumerate(train_data):

  tokenized_sentence, token_idxs, correct_label = example['input'], example['noun_tok_poses'], example['class']

  ideal_dist = torch.Tensor(dirac_mass(correct_label)).to(device=DEVICE)
  predicted = cls([token_idxs, tokenized_sentence])

  print(predicted, ideal_dist)
  loss += loss_function(predicted, ideal_dist)

  if index % 64 == 0: #Can we improve this naive batching strategy?
    loss.backward() #loss is a tensor and contains a backpropagation method
    optimizer.step() #step and then zero out
    optimizer.zero_grad()
    loss = 0

  #monitor performance
  if index % 1000 == 0:
    loss, accuracy = compute_validation_loss(cls)
    print(index, loss.item(), accuracy)