import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig
import tqdm

DEVICE = torch.device('cuda')
CONFIG = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states = True)

BERTHSSIZE = 768
MAXTOKS = 1
NUM_CLASSES=20

CLASS = "name1" #gender
#CLASS = "name2" #proximity

girls_names = [
    "Emily",
    "Sophia",
    "Olivia",
    "Ava",
    "Isabella",
    "Mia",
    "Amelia",
    "Harper",
    "Evelyn",
    "Abigail",
    "Charlotte",
    "Emma",
    "Scarlett",
    "Grace",
    "Lily",
    "Chloe",
    "Aria",
    "Ella",
    "Madison",
    "Zoe"
]

guys_names = [
    "Liam",
    "Noah",
    "William",
    "James",
    "Oliver",
    "Benjamin",
    "Elijah",
    "Lucas",
    "Mason",
    "Logan",
    "Alexander",
    "Ethan",
    "Jacob",
    "Michael",
    "Daniel",
    "Henry",
    "Jackson",
    "Sebastian",
    "Aiden",
    "Matthew"
]


if CLASS == "name1":
  SET = guys_names
elif CLASS == "name2":
  SET = girls_names

import json

datafile = open("train_data.json", "r")
dataset = json.load(datafile)

def dirac_mass(classes, cat):
  ret = [0]*len(classes)
  ret[classes.index(cat)] = 1
  return ret

#Largely borrowed from April 4th Colab notebook! Cited in paper.
class BERTHiddenStateClassifier(nn.Module):

  def __init__(self, layer):
    super(BERTHiddenStateClassifier, self).__init__()

    self.layer = layer
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # from https://huggingface.co/docs/transformers/model_doc/bert#tfbertmodel
    # now "bert(**input)" will contain a key for all hidden layers (13)
    self.bert = BertModel.from_pretrained("bert-base-uncased", config = CONFIG).to(device=DEVICE)

    self.W = nn.Linear(BERTHSSIZE*MAXTOKS, NUM_CLASSES)
    self.Softmax = nn.Softmax()

  # vector, it should return the output
  def forward(self, inp):

    #encode and move to GPU
    token_idx = inp[0]
    encoded_input = self.tokenizer(inp[1], return_tensors = "pt")

    for key, value in encoded_input.items():
      encoded_input[key] = value.to(device = DEVICE)

    emb = self.bert(**encoded_input)["hidden_states"][self.layer][0][token_idx].detach()

    outp = self.Softmax(self.W(emb))
    return outp

  def predict(self, noun_tokens, sentence): #classify and generate prediction to compute loss
    prob = self.forward([noun_tokens,sentence])
    return torch.topk(prob, 1)[1].item()

loss_function = nn.CrossEntropyLoss()
def compute_validation_loss(model, set):

  total_loss = 0
  total_correct = 0
  count_examples = len(set)

  for example in tqdm.tqdm(set):

    tokenized_sentence, token_idxs, correct_label = example['input'], example['pron_tok_pos'], example[CLASS]

    ideal_dist = torch.Tensor(dirac_mass(SET, correct_label)).to(device=DEVICE)
    predicted = model([token_idxs, tokenized_sentence])
    total_loss += loss_function(predicted, ideal_dist)
    total_correct += int(SET[torch.topk(predicted, 1)[1].item()] == correct_label)

  return total_loss / count_examples, total_correct / count_examples

if __name__=='__main__':
  ITER = 6
  cls = BERTHiddenStateClassifier(ITER).to(device=DEVICE)
  optimizer = torch.optim.Adam(cls.parameters(), lr=0.002)

  loss = 0
  for index, example in enumerate(dataset['train']):

    tokenized_sentence, token_idxs, correct_label = example['input'], example['pron_tok_pos'], example[CLASS]

    ideal_dist = torch.Tensor(dirac_mass(SET, correct_label)).to(device=DEVICE)
    predicted = cls([token_idxs, tokenized_sentence])
    loss += loss_function(predicted, ideal_dist)

    if index % 64 == 0: #Can we improve this naive batching strategy?
      loss.backward() #loss is a tensor and contains a backpropagation method
      optimizer.step() #step and then zero out
      optimizer.zero_grad()
      loss = 0

    #monitor performance
    if index % 1000 == 0:
      loss, accuracy = compute_validation_loss(cls, dataset["validation"])
      print(index, loss.item(), accuracy)

  _,final_accuracy = compute_validation_loss(cls, dataset['test'])
  print("ACCURACY",ITER,final_accuracy)