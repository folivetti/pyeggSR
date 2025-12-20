"""
Test suite for the evaluate function.
This demonstrates how to use the evaluate function to evaluate expression trees.
"""

import numpy as np
import pandas as pd
from expr import *
from egraph import *
from evaluate import *
from readimage import _read_inputs, _read_mask
import cv2 
import random
from dataclasses import dataclass, fields, is_dataclass
from copy import copy 

DIM = 6  # number of input channels
cache_fitness = {}
cache_params = {}
count = 0 

class MCTSNode:
    def __init__(self, eids):
        self.eids = eids
        self.children = []
        self.visits = 0
        self.rewards = []

    def uct_score(self, total_simulations, exploration_param=1.41):
        if self.visits == 0:
            return float('inf')
        exploitation = sum(self.rewards) / self.visits
        exploration = exploration_param * np.sqrt(np.log(total_simulations) / self.visits)
        return exploitation + exploration

# The e-graph will be a global variable HA HA HA 
egraph = EGraph()
eids = []
for i in range(DIM):
    eid = egraph.add(Var(i))
    eids.append(eid)

N_TERMS = DIM
N_CHILDREN = 10
MAX_NODES = 30 
P_OUT = 0.2

_, operators = get_terminals_and_operators(Expr)

root = MCTSNode(eids)
newnode = MCTSNode([])
newnode.visits = 1
newnode.rewards = [0.3]

def select(node):
    current_node = node
    path = []
    eids = node.eids
    while current_node.children and len(path) < MAX_NODES-3:
        total_simulations = sum(child.visits for child in current_node.children)
        uct_scores = [child.uct_score(total_simulations) for child in current_node.children]
        best_child_index = np.argmax(uct_scores)
        current_node = current_node.children[best_child_index]
        path.append(best_child_index)
        eids += current_node.eids
    return current_node, path, eids

def expansion(node, eids):
    global egraph 

    is_new = len(eids) - N_TERMS >= MAX_NODES - 2
    new_eid = []
    while len(node.children) < N_CHILDREN:
        op = np.random.choice(operators)
        field_values = {}
        for f in fields(op):
            field_values[f.name] = np.random.choice(eids)
        new_op = op(**field_values)
        # check if new_op is new 
        new_eid = egraph.add(new_op)
        if new_eid not in [eid for child in node.children for eid in child.eids]:
            new_node = MCTSNode([new_eid])
            node.children.append(new_node)
            is_new = True
            new_eid = [new_eid]
        else:
            print("Expansion generated existing node, retrying...")
    return node, eids + new_eid

def simulation(node, eids, imgs, masks):
    global egraph 

    eids_copy = eids.copy()
    is_out = (len(eids_copy) - N_TERMS) >= MAX_NODES - 1
    new_eid = eids[-1]
    while not is_out:
        op = np.random.choice(operators)
        field_values = {}
        for f in fields(op):
            field_values[f.name] = np.random.choice(eids)
        new_op = op(**field_values)
        new_eid = egraph.add(new_op)

        is_out = (len(eids_copy) - N_TERMS) >= MAX_NODES or random.random() < P_OUT
        if is_out and new_eid not in cache_fitness:
            eids_copy.append(new_eid)
        else:
            is_out = False
    root = new_eid
    score = calc_score(egraph, root, imgs, masks)
    #print("LEN: ", len(eids_copy)-N_TERMS, " SCORE: ", score)
    return score, root

def get_enode(egraph, eid):
    return next(iter(egraph.map_class[eid].enodes))

def get_n_params(egraph, root):
    enode = next(iter(egraph.map_class[egraph.find(root)].enodes))
    c = number_of_consts(enode)
    for child in children(enode):
        c += get_n_params(egraph, child)
    return c

def fitness(eg, eclass, params, image, target):
    result, _ = evaluate_egraph(eclass, eg, params, image, False)
    iou = compute_iou(target, result)
    return iou

def calc_score(egraph, root, imgs, masks):
    global count

    if root in cache_fitness:
        return cache_fitness[root]
    n_params = get_n_params(egraph, root)

    best_params = None 
    best_score = 0.0 

    for _ in range(1):
        params = np.random.randint(0, 255, size=(n_params,)).tolist()
        score = 0.0
        for img, mask in zip(imgs, masks):
            score += fitness(egraph, root, params, img, mask)
        score /= len(imgs)
        if score > best_score:
            best_score = score
            best_params = copy(params)
    #scores = [fitness(egraph, root, params, img, mask) for img, mask in zip(imgs, masks)]
    cache_fitness[root] = best_score
    cache_params[root] = best_params
    count = count + 1
    return score

def backpropagation(score, root, path):
    n = root
    update(n, score)
    for idx in path:
        n = n.children[idx]
        update(n, score)
def update(node, score):
    node.visits += 1
    node.rewards.append(score)

def mcts(imgs, masks, n_iterations=1000):
    global root 

    best_score = 0.0
    for it in range(n_iterations):
        # Selection
        selected_node, path, eids = select(root)
        # Expansion
        expanded_node, new_eids = expansion(selected_node, eids)
        # Simulation
        score, r = simulation(expanded_node, new_eids, imgs, masks)
        # Backpropagation
        backpropagation(score, root, path)

        if score > best_score:
            best_score = score
            print(f"New best score: {best_score}")
            print(">>", r, showEGraph(egraph, r))
        if it % 500 == 0:
            print(f"Iteration {it}, best score so far: {best_score}")

def calc_test_score(egraph, imgs, masks):
    # root is the key of the greatest value in cache_fitness 
    score = 0.0
    for i in range(len(imgs)):
        root = max(cache_fitness, key=lambda k: cache_fitness.get(k))
        params = cache_params[root]
        score += fitness(egraph, root, params, imgs[i], masks[i])
    return score / len(imgs)


def load_datasets(dirname):
    df = pd.read_csv(f"datasets/{dirname}/dataset.csv", sep=";")
    imgs = [_read_inputs(f"datasets/{dirname}/{v}")[0] for v in df[df.set == "training"].input.to_list()]
    masks = [_read_mask(f"datasets/{dirname}/{v}", imgs[i][0].shape) for i,v in enumerate(df[df.set == "training"]["label"].to_list())]

    test_imgs = [_read_inputs(f"datasets/{dirname}/{v}")[0] for v in df[df.set == "testing"].input.to_list()]
    test_masks = [_read_mask(f"datasets/{dirname}/{v}", test_imgs[i][0].shape) for i,v in enumerate(df[df.set == "testing"]["label"].to_list())]

    return imgs, masks, test_imgs, test_masks

def test_mcts():
    global count 

    imgs, masks, test_imgs, test_masks = load_datasets("melanoma")
    mcts(imgs, masks, n_iterations=1000)
    test_score = calc_test_score(egraph, test_imgs, test_masks)
    print(f"Final test score: {test_score} with {count} evaluations")

if __name__ == "__main__":
    
    try:
        test_mcts()
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
