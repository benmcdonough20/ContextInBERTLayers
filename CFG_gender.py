import random
import json
from transformers import BertTokenizer
import tqdm
import random

#Little rhyming grammar I made

terminals_mammals = [
    ("dog","cat","rat","horse","tapir","pig","cow","bear","elephant","squirrel","mouse","vole","lemming","capybara","baboon","ape","chinchilla","aardvaark","platypus","koala"),
    ("squatted","rolled","twirled","flew","darted","meandered","scampered","lumbered","bounded","pounced","dug","planked","loped"),
    ("gruff","hairy","miniscule","plump","pernicious","skinny","bored","unamused","inquisitive","ardent","infatuated","frustrated","upset","resigned"),
    ("below", "under", "close to", "beside", "next to")
]

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
        'NP': [('Det N', 'NP PP'), (.8,.2)],
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

    def bookend_nouns(self):
        leaves = self.rootnode.find_leaves()
        i = 0
        while leaves[i].parent.word != 'N':
            i += 1
        j = len(leaves)-1
        while leaves[j].parent.word != 'N':
            j -= 1
        return(leaves[i], leaves[j])
    
    def add_pronoun(self, pronoun):
        replacement = EmptyNode(pronoun, self.grammar)
        replacement.parent = self
        self.rootnode.leaves[0] = replacement

    def __str__(self):
        return str(self.rootnode)

    def tokenize(self): #Omitting CLS and SEP tokens
        return concat(*self.rootnode.tokenize())

    def __hash__(self):
        return self.__str__().__hash__()

#tree nodes
class Node:
    def __init__(self, word, parent):
        self.parent = parent #backpointer
        self.grammar = parent.grammar
        self.word = word
        self.leaves = []
        self.tokens = []
        self.subtree_contains_noun = False
        if word in self.grammar:
            option = random.choices(self.grammar[word][0], weights = self.grammar[word][1])[0]
            self.leaves = [Node(leaf, self) for leaf in option.split(" ")]
        else:
            self.tokens = tokenizer(word)['input_ids'][1:-1] #omit CLS and SEP tokens

    def __str__(self):
        if len(self.leaves) == 0:
            return self.word
        return " ".join(map(str, self.leaves))

    def find_leaves(self):
        if len(self.leaves) == 0:
            return [self]
        return concat(*[n.find_leaves() for n in self.leaves])

    def tokenize(self):
        if not self.leaves:
            return [self.tokens]
        return concat(*[l.tokenize() for l in self.leaves])

    def first_left_subtree(self):
        if len(self.parent.leaves) == 2 and self.parent.leaves[1] == self:
            return self.parent.leaves[0]
        else:
            return self.parent.first_left_subtree()

    def position(self):
        return len(str(self.first_left_subtree()).split(" "))

    def token_positions(self):
        start_idx = len(self.first_left_subtree().tokenize()) #remember to account for CLS
        num_toks = len(self.tokens)
        return [start_idx + i for i in range(num_toks)]

    def get_phrase(self, branches, direction):
        nouns = [l for l in self.leaves if l.word in branches]
        if len(nouns) == 0:
            return self.leaves[direction]
        else:
            return nouns[direction].get_phrase(branches)

class EmptyNode:
    def __init__(self, word, grammar):
        self.word = word
        self.leaves = []
        self.grammar = grammar
        self.tokens = tokenizer(word)['input_ids'][1:-1] #omit CLS and SEP tokens

    def __str__(self):
        return self.word

    def tokenize(self):
        return [self.tokens]



"""
Generate Jack/Jill sentences with attractors
"""
def gen_gender_sentences(num):
    sentences = []
    for i in tqdm.tqdm(range(num)):
        grammar = gen_grammar(terminals_mammals)
        noun1 = noun2 = EmptyNode("N", grammar)
        s = Sentence(grammar)
        while noun1.word == noun2.word:
            s = Sentence(grammar)
            noun1, noun2 = s.bookend_nouns()

        name1 = random.choice(guys_names)
        name2 = random.choice(girls_names)
        pronoun = "he"

        par1 = noun1.parent
        while par1.word != "NP":
            par1 = par1.parent
        par1.leaves = [EmptyNode(name1, grammar)]

        par2 = noun2.parent
        while par2.word != "NP":
            par2 = par2.parent
        par2.leaves = [EmptyNode(name2, grammar)]

        sentence2 = Sentence(grammar)
        sentence2.add_pronoun(pronoun)
        sentence_tot = str(s)+". " + str(sentence2)+"."
        sentences.append(
            {
                "input":sentence_tot,
                "name1":name1,
                "name2":name2,
                "pron_tok_pos":len(s.tokenize())+2
            }
    )
    return sentences


with open("train_data.json", "w") as f:
    f.write(json.dumps({
        'train':gen_gender_sentences(70000),
        'validation':gen_gender_sentences(2000),
        'test':gen_gender_sentences(20000)
        },
        indent = 2))