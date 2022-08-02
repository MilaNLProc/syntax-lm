from allennlp_models.pretrained import load_predictor
import numpy as np
import torch

PoS_tags: dict = {
    'X': 0, 'ADJ': 1, 'NOUN': 2, 'PUNCT': 3,
    'AUX': 4, 'PRON': 5, 'PART': 6, 'VERB': 7,
    'ADP': 8, 'PROPN': 9, 'DET': 10, 'CCONJ': 11,
    'ADV': 12, 'SCONJ': 13, 'INTJ': 14, 'SYM': 15,
    'NUM': 16
}

#get position of a certain key in a list (multiple positions if multiple values)
def get_key(val, par):
    keys = list()
    for i in range(len(par)):
        if val == par[i]:
            keys.append(i)

    return keys

#recursion to visit the tree (ok since trees are little)
def rec_tree(tree, vect, idx):
    children = get_key(idx, vect)
    tree.append(idx)

    for child in children:
        rec_tree(tree, vect, child)

    return

#treat position of the parents
def conv_parents(par, idx):
    new_par = list()
    for p in par:
        if p == -1:
            new_par.append(-1)
        else:
            new_par.append(idx.index(p))

    return new_par

#method to divide trees in same tweet in different lists
def divide_intraphrase_trees(words, poss, parents):
    roots = get_key(-1, parents) #take roots of the tweet
    words_root = list()
    poss_root = list()
    parents_root = list()
    for root in roots:
        tree = list()
        rec_tree(tree, parents, root) #call recursevely
        tree.sort() #retrieve indeces of the words in the tree
        words_root.append((np.array(words)[np.array(tree).astype(int)]).tolist()) #extract words associated to the tree
        poss_root.append((np.array(poss)[np.array(tree).astype(int)]).tolist()) #extract  pos tags associated to the tree
        new_par = (np.array(parents[i])[np.array(tree).astype(int)]).tolist() #extract parents associated to the tree
        parents_root.append(conv_parents(new_par, tree)) #convert indeces of parents from tweet "referment" to tree referment

    return words_root, poss_root, parents_root


#function to retrieve order of visit a tree (DFS)
def visit_single_tree(par, parents):
    children = get_key(par, parents)
    visit_order = list()
    parent_visit_order = list()
    #one vector for node to visit, one vector for the parents
    for child in reversed(children):
        visit_order.append(child)
        parent_visit_order.append(par)
    #find next nodes to visit
    for child in reversed(children):
        v, p = visit_single_tree(child, parents)
        visit_order = visit_order + v
        parent_visit_order = parent_visit_order + p

    return visit_order, parent_visit_order


def visit_tree(parents):
    visit_order = list()
    parent_visit_order = list()
    for p in parents:
        vo, pvo = visit_single_tree(-1, p)
        visit_order.append(vo)
        parent_visit_order.append(pvo)

    return visit_order, parent_visit_order

# preprocessing for pad purposing
def preprocess(words, poss, visit_order, parent_visit_order, size, max_root):
    words_string = list()
    pad_flag = list()
    orig_len = len(words)
    pad_trees = max_root - orig_len
    for p in range(pad_trees): #we want "padding trees" at the beginning of the tree sequence of each tweets
        words.insert(0, [" "])
        poss.insert(0, [])
        visit_order.insert(0, [])
        parent_visit_order.insert(0, [])
        pad_flag.append([0])
    for p in range(orig_len):
        pad_flag.append([1])

    for i in range(len(poss)):
        w = ' '.join(words[i])
        words_string.append(w)
        pad = size - len(poss[i])
        for j in range(pad):
            poss[i].append(25) #25 pad value for POS tags
            visit_order[i].append(-2) #-2 pad value for visit ordering
            parent_visit_order[i].append(-2) #-2 pad value for parents of visit ordering

    return words_string, poss, visit_order, parent_visit_order, pad_flag


class DepParser():

    def __init__(self):
        self.predictor = load_predictor("structured-prediction-biaffine-parser")

    def parsing(self, sentence):
        phrase = list()
        pos = list()
        parents = list()
        preds = self.predictor.predict(sentence)
        words = preds["words"]
        poss = preds["pos"]
        deps = preds["predicted_dependencies"]
        #PUT HERE RANDOM STUFF IN FUTURE
        heads = preds["predicted_heads"]

        for word, pos, dep, head in zip(words, poss, deps, heads):
            phrase.append(word)
            parents.append(int(head) - 1)
            pos.append(PoS_tags[pos])

        words, poss, parents = divide_intraphrase_trees(phrase, pos, parents)
        visit_order, parent_visit_order = visit_tree(parents)  # take vectors for order of visiting

        words, poss, visit_order, parent_visit_order, pad_mask_trees = preprocess(words, poss, visit_order, parent_visit_order, 20, 1)

        return {
            'words': words,
            'poss': poss,  # POS tags
            'visit_order': visit_order,  # extract visit order
            'parent_visit_order': parent_visit_order,  # parent of node visited
            'pad_mask_trees': pad_mask_trees  # pad mask for trees
        }

    def map_tokens(self, list_tokens_bert):
        list_map_tokbert_to_tokparse = list()
        list_map_attention = list()
        list_divisors = list()
        list_map_tree = list()
        for tokens_bert in list_tokens_bert:
            idx_tok = -1
            tree_idx = 0
            map_tree = list()
            map_tokbert_to_tokparse = list()
            # print(self.tokenizer.convert_ids_to_tokens(encoding['input_ids'].flatten()))
            for s in tokens_bert:
                if s != "<s>" and s != "</s>" and s != "<pad>":
                    if s.startswith('Ġ'):
                        idx_tok = idx_tok + 1
                    map_tokbert_to_tokparse.append(int(idx_tok))
                    map_tree.append(tree_idx)
                else:
                    map_tokbert_to_tokparse.append(int(-1))

            divisors = [[1]] * len(map_tokbert_to_tokparse)
            map_attention = [[0]] * len(map_tokbert_to_tokparse)

            for d in range(len(map_tokbert_to_tokparse)):
                if map_tokbert_to_tokparse[d] >= 0:
                    map_attention[map_tokbert_to_tokparse[d]] = [1]
                    if map_tokbert_to_tokparse[d - 1] == map_tokbert_to_tokparse[d]:
                        divisors[map_tokbert_to_tokparse[d]] = [divisors[map_tokbert_to_tokparse[d]][0] + 1]

            list_map_tokbert_to_tokparse.append(map_tokbert_to_tokparse)
            list_map_attention.append(map_attention)
            list_divisors.append(divisors)
        '''
        list_map_tokbert_to_tokparse = list()
        list_map_attention = list()
        list_divisors = list()

        for tokens_bert in list_tokens_bert:
            idx_tok = -1
            map_tokbert_to_tokparse = list()
            # print(self.tokenizer.convert_ids_to_tokens(encoding['input_ids'].flatten()))
            for s in tokens_bert:
                if s != "<s>" and s != "</s>" and s != "<pad>":
                    if s.startswith('Ġ'):
                        idx_tok = idx_tok + 1
                    map_tokbert_to_tokparse.append(int(idx_tok))

                else:
                    map_tokbert_to_tokparse.append(int(-1))

            divisors = [[1]] * len(map_tokbert_to_tokparse)
            map_attention = [[0]] * len(map_tokbert_to_tokparse)

            for d in range(len(map_tokbert_to_tokparse)):
                if map_tokbert_to_tokparse[d] >= 0:
                    map_attention[map_tokbert_to_tokparse[d]] = [1]
                    if map_tokbert_to_tokparse[d - 1] == map_tokbert_to_tokparse[d]:
                        divisors[map_tokbert_to_tokparse[d]] = [divisors[map_tokbert_to_tokparse[d]][0] + 1]

            list_map_tokbert_to_tokparse.append(map_tokbert_to_tokparse)
            list_map_attention.append(map_attention)
            list_divisors.append(divisors)
            '''
        return {
            'map_tokbert_to_tokparse': list_map_tokbert_to_tokparse,
            'divisors': list_divisors,
            'map_attention': list_map_attention  # extract visit order
        }