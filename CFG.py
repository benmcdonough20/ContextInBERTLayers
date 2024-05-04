import random
import json

#Little rhyming grammar I made
sample_grammar = {
    'N': [('mat', 'cat', 'rat'), (0.2, .6, .2)],
    'AP' : [('fat','flat'), (.5, .5)],
    'V': [('sat', 'spat'), (0.3, 0.4)],
    'VP': [('V', 'V PP'), (0.2, .8)],
    'PP': [('P NP',), (1.0,)],
    'P': [('on', 'behind', 'under'), (0.33, 0.33, 0.33)],
    'NP': [('Det N', 'NP PP'), (.8,.1)],
    'Det': [('the AP','the'), (.5, .5)],
    'S': [('NP VP',), (1.0,)]
} 

#store sentence and parse structure
class Sentence:
    def __init__(self):
        self.rootnode = Node("S")
    
    def get_main_phrase(self, branches):
        return self.get_phrase(self.rootnode, branches)

    def get_phrase(self, node, branches):
        nouns = [l for l in node.leaves if l.word in branches]
        if len(nouns) == 0:
            return node.leaves[0]
        else:
            return self.get_phrase(nouns[0], branches)
    
    def __str__(self):
        return str(self.rootnode)

#tree nodes
class Node:
    def __init__(self, word):
        self.word = word
        self.leaves = []
        if word in sample_grammar:
            option = random.choices(sample_grammar[word][0], weights = sample_grammar[word][1])[0]
            self.leaves = [Node(leaf) for leaf in option.split(" ")]
        
    def __str__(self):
        if len(self.leaves) == 0:
            return self.word
        return " ".join(map(str, self.leaves))

#generate a bunch of pairs with location of subject and verb
responses = []
for i in range(10):
    s = Sentence()
    subj = s.get_main_phrase(["N", "NP"]).word
    subj_pos = str(s).split(" ").index(subj)
    responses.append(
        {"input" : str(s),
        "subj_pos" : subj_pos,
        "class" : s.get_main_phrase(["V", "VP"]).word}
    )

with open("catmat_CFG_data.json", "w") as f:
    f.write(json.dumps(responses, indent = 4))