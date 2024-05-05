import random
import json

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
    ("big","small","angry","happy","disgruntled","nervous","mean","amiable","wild","flat","spiny","slender","sparkly","flighty","passionate","aggressive")
]

terminals_mammals = [
    ("dog","cat","rat","horse","tapir","pig","cow","bear","elephant","squirrel","mouse","vole","lemming","capybara","baboon","ape","chinchilla","aardvaark","platypus","koala"),
    ("squatted","rolled","twirled","flew","darted","meandered","scampered","lumbered","bounded","pounced","dug","planked","loped"),
    ("gruff","hairy","miniscule","plump","pernicious","skinny","bored","unamused","inquisitive","ardent","infatuated","frustrated","upset","resigned"),
]


def gen_grammar(terminals):
    nouns,verbs,adjectives = terminals
    return {
        'N': [('A N',)+nouns, (.2,)+(.8/len(nouns),)*len(nouns)],
        'A' : [adjectives, (1/len(adjectives),)*len(adjectives)],
        'V' : [verbs, (1/len(verbs),)*len(verbs)],
        'VP': [('V', 'V PP'), (0.2, .8)],
        'PP': [('P NP',), (1.0,)],
        'P': [('on', 'behind','over','around','under', 'in front of'), (1/6,)*6],
        'NP': [('Det N', 'NP PP'), (.8,.1)],
        'Det': [('the',), (1.0,)],
        'S': [('NP VP',), (1.0,)]
    }

#store sentence and parse structure
class Sentence:
    def __init__(self, grammar):
        self.rootnode = Node("S", grammar)
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

#tree nodes
class Node:
    def __init__(self, word, grammar):
        self.grammar = grammar
        self.word = word
        self.leaves = []
        if word in grammar:
            option = random.choices(grammar[word][0], weights = grammar[word][1])[0]
            self.leaves = [Node(leaf, self.grammar) for leaf in option.split(" ")]

    def __str__(self):
        if len(self.leaves) == 0:
            return self.word
        return " ".join(map(str, self.leaves))

class EmptyNode:
    def __init__(self, word):
        self.word = word
        self.leaves = []

    def __str__(self):
        return self.word

#generate a bunch of pairs with location of subject and verb
responses = []
for i in range(100):
    s = Sentence(gen_grammar(terminals_fish))
    subj = s.get_main_phrase(["V", "VP"]).word
    subj_pos = str(s).split(" ").index(subj)
    responses.append(
        {"input" : str(s),
        "subj_pos" : subj_pos,
        "class" : s.get_main_phrase(["N", "NP"]).word}
    )

with open("catmat_CFG_data.json", "w") as f:
    f.write(json.dumps(responses, indent = 4))