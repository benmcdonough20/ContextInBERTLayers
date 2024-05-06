import random
import json
from transformers import BertTokenizer
import tqdm

#Little rhyming grammar I made
sample_grammar = {
    'N': [('mat', 'cat', 'rat'), (0.5, .5, .5)],
    'AP' : [('fat','flat'), (.5, .5)],
    'V': [('sat', 'spat'), (0.5, 0.5)],
    'VP': [('V', 'V PP'), (0.3, .7)],
    'PP': [('P NP',), (1.0,)],
    'P': [('on', 'behind', 'under'), (0.33, 0.33, 0.33)],
    'NP': [('Det N', 'NP PP'), (.7,.3)],
    'Det': [('the AP','the'), (.5, .5)],
    'S': [('NP VP',), (1.0,)]
} 

terminals_fish = [
    ("salmon","trout","tuna","cod","halibut","bass","catfish","swordfish","mackerel","haddock","sardine","carp","mahi-mahi","perch","snapper","flounder","grouper","barracuda","angelfish","clownfish"),
    ("jumped", "sat", "danced", "swam", "ran", "galloped", "crawled", "slithered", "spat", "flopped", "stood", "hopped", "knocked", "layed", "knelt", "foundered", "spat", "plopped"),
    ("big","small","angry","happy","disgruntled","nervous","mean","amiable","wild","flat","spiny","slender","sparkly","flighty","passionate","aggressive"),
    ("on top of", "over", "around", "with", "by")
]

terminals_mammals = [
    ("dog","cat","rat","horse","tapir","pig","cow","bear","elephant","squirrel","mouse","vole","lemming","capybara","baboon","ape","chinchilla","aardvaark","platypus","koala"),
    ("squatted","rolled","twirled","flew","darted","meandered","scampered","lumbered","bounded","pounced","dug","planked","loped"),
    ("gruff","hairy","miniscule","plump","pernicious","skinny","bored","unamused","inquisitive","ardent","infatuated","frustrated","upset","resigned"),
    ("below", "under", "close to", "beside", "next to")
]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def concat(*arrs):
    ret = []
    for arr in arrs:
        ret += arr
    return ret

def gen_grammar(terminals):
    nouns,verbs,adjectives,prepositions = terminals
    return {
        'N': [('A N',)+nouns, (.2,)+(.8/len(nouns),)*len(nouns)],
        'A' : [adjectives, (1/len(adjectives),)*len(adjectives)],
        'V' : [verbs, (1/len(verbs),)*len(verbs)],
        'VP': [('V', 'V PP'), (0.2, .8)],
        'PP': [('P NP',), (1.0,)],
        'P': [prepositions, (1/len(prepositions),)*len(prepositions)],
        'NP': [('Det N', 'NP PP'), (.8,.1)],
        'Det': [('the','a'), (0.5,0.5)],
        'S': [('NP VP',), (1.0,)]
    }

#store sentence and parse structure
class Sentence:
    def __init__(self, grammar):
        self.rootnode = Node("S", EmptyNode("CLS", grammar))
        self.grammar = grammar
    
    def get_main_phrase(self, branches):
        return self.get_phrase(self.rootnode, branches)

    def get_phrase(self, node, branches):
        nouns = [l for l in node.leaves if l.word in branches]
        if len(nouns) == 0:
            return node.leaves[0]
        else:
            return self.get_phrase(nouns[0], branches)
    
    def add_pronoun(self, pronouns):
        replacement = EmptyNode(random.choice(pronouns))
        self.rootnode.leaves[0] = replacement

    def __str__(self):
        return str(self.rootnode)

    def tokenize(self):
        return concat([101],self.rootnode.tokenize(),[102])

#tree nodes
class Node:
    def __init__(self, word, parent):
        self.parent = parent #backpointer
        self.grammar = parent.grammar
        self.word = word
        self.leaves = []
        self.tokens = []
        if word in self.grammar:
            option = random.choices(self.grammar[word][0], weights = self.grammar[word][1])[0]
            self.leaves = [Node(leaf, self) for leaf in option.split(" ")]
        else:
            self.tokens = tokenizer(word)['input_ids'][1:-1] #omit CLS and SEP tokens

    def __str__(self):
        if len(self.leaves) == 0:
            return self.word
        return " ".join(map(str, self.leaves))

    def tokenize(self):
        if len(self.leaves) == 0:
            return self.tokens
        return concat(*[l.tokenize() for l in self.leaves])

    def first_left_subtree(self):
        if len(self.parent.leaves) == 2 and self.parent.leaves[1] == self:
            return self.parent.leaves[0]
        else:
            return self.parent.first_left_subtree()

    def position(self):
        return len(str(self.first_left_subtree()).split(" "))

    def token_positions(self):
        start_idx = len(self.first_left_subtree().tokenize()) + 1 #to account for CLS
        num_toks = len(self.tokens)
        return [start_idx + i for i in range(num_toks)]

class EmptyNode:
    def __init__(self, word, grammar):
        self.word = word
        self.leaves = []
        self.grammar = grammar

    def __str__(self):
        return self.word


def gen(terminals, num):
    ret = []
    sentences = set()
    grammar = gen_grammar(terminals)
    for i in tqdm.tqdm(range(num)):
        s = Sentence(grammar)
        sentences.add(str(s))
        noun = s.get_main_phrase(["N", "NP"])
        noun_tok_pos = noun.token_positions()
        ret.append(
            {
                "input" : str(s),
                "noun_tok_poses" : list(noun_tok_pos),
                "class" : noun.word
            }
        )
    print("Unique sentences:", len(list(sentences)), len(list(sentences))/num)
    return ret

with open("train_data.json", "w") as f:
    f.write(json.dumps({
        'train':gen(terminals_mammals, 10000),
        'validation':gen(terminals_mammals, 800),
        'test':gen(terminals_fish, 2000)
        },
        indent = 2))