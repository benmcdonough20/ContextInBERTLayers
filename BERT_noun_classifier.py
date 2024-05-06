import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig
import tqdm

DEVICE = torch.device('cuda')
CONFIG = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states = True)

BERTHSSIZE = 768
MAXTOKS = 3

import json

datafile = open("/home/ben/Documents/Repos/ContextInBERTLayers/train_data.json", "r")
dataset = json.load(datafile)

classes = list(set([t["class"] for t in dataset['train']]))

def dirac_mass(cat):
  ret = [0]*len(classes)
  ret[classes.index(cat)] = 1
  return ret

class BERTHiddenStateClassifier(nn.Module):

  def __init__(self, layer):
    super(BERTHiddenStateClassifier, self).__init__()

    self.layer = layer
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # from https://huggingface.co/docs/transformers/model_doc/bert#tfbertmodel
    # now "bert(**input)" will contain a key for all hidden layers (13)
    self.bert = BertModel.from_pretrained("bert-base-uncased", config = CONFIG).to(device=DEVICE)
    """
    self.W1 = nn.Linear(BERTHSSIZE, len(classes))
    self.W2 = nn.Linear(BERTHSSIZE, len(classes))
    self.W3 = nn.Linear(BERTHSSIZE, len(classes))
    self.W4 = nn.Linear(BERTHSSIZE, len(classes))
    """
    self.W = nn.Linear(BERTHSSIZE*MAXTOKS, len(classes))
    self.Softmax = nn.Softmax()

    """
    replace this with RNN! 
    """
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
    """

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
    embs = self.bert(**encoded_input)["hidden_states"][self.layer][0]
    ret = [embs[i].detach() for i in tokens_idxs[:MAXTOKS]]+[torch.zeros(BERTHSSIZE).to(DEVICE) for i in range(MAXTOKS-len(tokens_idxs))]
    """
    lin1 = self.W1(ret[0])
    lin2 = self.W2(ret[1])
    lin3 = self.W3(ret[2])
    lin4 = self.W4(ret[3])
    outp = self.Softmax(lin1+lin2+lin3+lin4)
    """
    outp = self.Softmax(self.W(torch.cat(ret)))
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

    tokenized_sentence, token_idxs, correct_label = example['input'], example['noun_tok_poses'], example['class']

    ideal_dist = torch.Tensor(dirac_mass(correct_label)).to(device=DEVICE)
    predicted = model([token_idxs, tokenized_sentence])
    total_loss += loss_function(predicted, ideal_dist)
    total_correct += int(classes[torch.topk(predicted, 1)[1].item()] == correct_label)

  return total_loss / count_examples, total_correct / count_examples

if __name__=='__main__':
  all_acuracies = []
  for hidden_state_iter in range(13):
    cls = BERTHiddenStateClassifier(hidden_state_iter).to(device=DEVICE)
    optimizer = torch.optim.Adam(cls.parameters(), lr=0.0005)

    loss = 0
    for index, example in enumerate(dataset['train']):

      tokenized_sentence, token_idxs, correct_label = example['input'], example['noun_tok_poses'], example['class']

      ideal_dist = torch.Tensor(dirac_mass(correct_label)).to(device=DEVICE)
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
    all_acuracies.append(compute_validation_loss(cls,dataset["test"][1]))
  print(all_acuracies)