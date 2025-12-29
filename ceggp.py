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
import cProfile 
import io 
import sys
import pstats
from pstats import SortKey
import optuna 

DIM = 6  # number of input channels
cache_fitness = {}
cache_params = {}
count = 0 
DEBUG = False 
P_INTRON = 0.1 # 0.25

def get_enode(egraph, eid):
    return next(iter(egraph.map_class[eid].enodes))

def fitness(eg, eclass, params, image, target, useCache=False):
    result, _ = evaluate_egraph(eclass, eg, params, image, useCache)
    eg.cache = {}
    iou = compute_iou(target, result)
    return iou

def create_operator(op, children):
    field_values = {}
    for f, c in zip(fields(op), children):
        field_values[f.name] = c 
    return op(**field_values)

def get_fst_rnd_egraph(size):
    terminals, operators = get_terminals_and_operators(Expr)
    egraph = EGraph()
    # insert all variables up to d
    eids = {}
    eids[0] = []
    for i in range(DIM):
        eid = egraph.add(Var(i))
        eids[0].append(eid)
    # choose an operator from Expr
    for h in range(1, size + 1):
        cs = np.random.choice([c for k, v in eids.items() for c in v], 2)
        op = create_operator(np.random.choice(operators), cs)
        eid = egraph.add(op)
        eids[h] = [eid]
    out = np.random.choice(size) + 1
    return egraph, eids, out 

def get_active_nodes(egraph, eids, out):
    active = set()
    inactive = set()
    active.add(out)
    for i in range(out+1, len(eids)):
        inactive.add(i)
    cs = children(get_enode(egraph, eids[out][0]))
    for i in range(out-1, 0, -1):
        eclass = eids[i][0]
        if eclass in cs:
            active.add(i)
            enode = get_enode(egraph, eclass)
            cs += children(enode)
        else:
            inactive.add(i)
    return list(active), list(inactive)

def check_novelty(egraph, old_eids, column, out, new_enode, active):
    if new_enode not in egraph.hashcon:
        return True 
    old_eid = old_eids[column][0]
    new_eid = egraph.hashcon[new_enode]

    replacements = {}
    replacements[old_eid] = new_eid 

    for d in range(column + 1, max(old_eids.keys()) + 1):
        if d not in active:
            continue
        
        enode = get_enode(egraph, old_eids[d][0])
        cs = children(enode)
        # replace any occurrence of eid with new_eid
        for i in range(len(cs)):
            if cs[i] in replacements:
                cs[i] = replacements[cs[i]]
        n = replaceChildren(enode, cs)
        if n not in egraph.hashcon:
            return True 
        old_eid = old_eids[d][0]
        new_eid = egraph.hashcon[n]
        replacements[old_eid] = new_eid 
        if d == out:
            if new_eid in cache_fitness:
                return False
            else:
                return True
    return False

def apply_change(egraph, old_eids, column, new_enode):
    eids = {k: v.copy() for k, v in old_eids.items()}
    eid = old_eids[column][0]
    new_eid = egraph.add(new_enode)
    eids[column] = [new_eid]

    replacements = {}
    replacements[eid] = new_eid 

    for d in range(column + 1, max(eids.keys()) + 1):
        enode = get_enode(egraph, eids[d][0])
        cs = children(enode)
        # replace any occurrence of eid with new_eid
        for i in range(len(cs)):
            if cs[i] in replacements:
                cs[i] = replacements[cs[i]]
        n = replaceChildren(enode, cs)
        updated_eid = egraph.add(n)
        if eids[d][0] != updated_eid:
            replacements[eids[d][0]] = updated_eid
        eids[d] = [updated_eid]
    return egraph, eids

def mutation(egraph, eids, out, atNode=True):
    terminals, operators = get_terminals_and_operators(Expr)
    active, inactive = get_active_nodes(egraph, eids, out)
    if random.random() < P_INTRON:
        column = np.random.choice(inactive)
        isActive = False
    else:
        column = np.random.choice(active)
        isActive = True
    eid = np.random.choice(eids[column])
    enode = get_enode(egraph, eid)
    cs = children(enode)
    csnew = [c for k, v in list(eids.items())[:column] for c in v]
    if atNode:
        cs += [np.random.choice(csnew)]  # add a new children in case the original node has only one and the replacement has two
    # choose a random operator
    candidates = []
    if atNode:
        candidates = [create_operator(op, cs) for op in operators]
    else:
        for c in csnew:
            for ix in range(len(cs)):
                cs_replace = copy(cs)
                cs_replace[ix] = c
                candidates.append(replaceChildren(enode, cs_replace))


    final_candidates = []

    for new_enode in candidates:
        if isActive: 
            if check_novelty(egraph, eids, column, out, new_enode, active):
                final_candidates.append(new_enode)
        else:
            if new_enode not in egraph.hashcon:
                final_candidates.append(new_enode)
    if len(final_candidates) == 0:
        return egraph, eids, column, out

    choice = np.random.choice(final_candidates)
    egraph, new_eids = apply_change(egraph, eids, column, choice)

    return egraph, new_eids, column, out
    

def mutate_node(egraph, eids, out):
    return mutation(egraph, eids, out, atNode=True)

def mutate_edge(egraph, eids, out):
    return mutation(egraph, eids, out, atNode=False)
    
def mutate_out(egraph, eids, out):
    out = np.random.choice(len(eids)-1)+1
    enode = get_enode(egraph, eids[out][0])
    if enode in cache_fitness and DEBUG:
        print("out in cache")
    return egraph, eids, out

def get_n_params(egraph, root):
    enode = next(iter(egraph.map_class[egraph.find(root)].enodes))
    c = number_of_consts(enode)
    for child in children(enode):
        c += get_n_params(egraph, child)
    return c

def calc_score(egraph, root, imgs, masks, recalc=False):
    global count, cache_fitness

    if root in cache_fitness and not recalc:
        return cache_fitness[root]
    n_params = get_n_params(egraph, root)

    for _ in range(1):
        best_params = None 
        best_score = 0.0 

        params = np.random.randint(0, 255, size=(n_params,)).tolist()
        # params = np.random.choice([10, 50, 100, 150, 200, 250], size=(n_params,)).tolist()
        score = 0.0
        for img, mask in zip(imgs, masks):
            score += fitness(egraph, root, params, img, mask, False)
        score /= len(imgs)
        if score > best_score:
            best_score = score
            best_params = copy(params)
        count = count + 1
    #scores = [fitness(egraph, root, params, img, mask) for img, mask in zip(imgs, masks)]
    cache_fitness[root] = best_score
    cache_params[root] = best_params
    clean_cache(egraph)
    return score

def calc_test_score(egraph, imgs, masks):
    # root is the key of the greatest value in cache_fitness 
    score = 0.0
    root = max(cache_fitness, key=lambda k: cache_fitness.get(k))
    params = cache_params[root]
    for i in range(len(imgs)):
        score += fitness(egraph, root, params, imgs[i], masks[i], False)
    return score / len(imgs)

def step(egraph, eids, out, imgs, masks, best_score, it, p_node = 0.25, p_edge = 0.5, p_accept = 0.0):
    global count
    orig_eids = {k: v.copy() for k, v in eids.items()}
    orig_out = out
    r = random.random()
    recalc = False 
    if r < p_node:
        if DEBUG:
            print("mut node")
        egraph, eids, d, out = mutate_node(egraph, eids, out)
    elif r < p_node + p_edge:
        if DEBUG:
            print("mut edge")
        egraph, eids, d, out = mutate_edge(egraph, eids, out)
    else:
        if DEBUG:
            print("mut out")
        egraph, eids, out = mutate_out(egraph, eids, out)
        recalc = True

    roots = [eids[out][0]] # [v[0] for k,v in eids.items() if len(v)>0 and k <= d]
    new_best = False
    for root in roots:
        old_count = count
        score = calc_score(egraph, root, imgs, masks, recalc)
        if score >= best_score and out != orig_out:
            if score >= best_score:
                if count > old_count:
                    print(f"{count},{score}")
                #print(f"New best score: {score} at iteration {it} with {count} evaluations")
                print(">>", root, showEGraph(egraph, root))
            #if score > best_score or egraph.map_class[egraph.find(root)].height <= egraph.map_class[egraph.find(max(cache_fitness, key=cache_fitness.get))].height:
                best_score = score
                new_best = True
    if score < best_score and np.random.random() >= p_accept:
        return egraph, orig_eids, best_score, orig_out
    return egraph, eids, best_score, out

def load_datasets(dirname):
    df = pd.read_csv(f"datasets/{dirname}/dataset.csv", sep=";")
    imgs = [_read_inputs(f"datasets/{dirname}/{v}") for v in df[df.set == "training"].input.to_list()]
    szs = [img[1] for img in imgs]
    imgs = [img[0] for img in imgs]
    masks = [_read_mask(f"datasets/{dirname}/{v}", szs[i]) for i,v in enumerate(df[df.set == "training"]["label"].to_list())]

    test_imgs = [_read_inputs(f"datasets/{dirname}/{v}", True) for v in df[df.set == "testing"].input.to_list()]
    szs_test = [img[1] for img in test_imgs]
    test_imgs = [img[0] for img in test_imgs]
    test_masks = [_read_mask(f"datasets/{dirname}/{v}", szs_test[i], True) for i,v in enumerate(df[df.set == "testing"]["label"].to_list())]

    return imgs, masks, test_imgs, test_masks

def test_evo(n_nodes, p_node, p_edge, n_evals, max_iter):
    global count
    imgs, masks, test_imgs, test_masks = load_datasets(sys.argv[1])

    best_score = 0.0
    egraph, eids, out = get_fst_rnd_egraph(n_nodes)
    roots = [eids[out][0]] # [v[0] for k,v in eids.items() if len(v)>0]
    for root in roots:
        score = calc_score(egraph, root, imgs, masks)
        print(f"score: {score}")
        if score >= best_score:
            print(">>", root, showEGraph(egraph, root))
            best_score = score

    # for iteration in range(1000):
    iteration = 0 
    #while count < 5000 and iteration < 5550:
    while count < n_evals and iteration < max_iter:
        egraph, eids, best_score, out = step(egraph, eids, out, imgs, masks, best_score, iteration, p_node, p_edge)
        iteration += 1
        if iteration % 1000 == 0:
            print(iteration, count)
    print(f"Final training score: {best_score}")
    test_score = calc_test_score(egraph, test_imgs, test_masks)
    print(f"Final test score: {test_score} with {count} evaluations")
    return best_score

def objective(trial):
    global cache_fitness, cache_params, count 
    cache_fitness = {}
    cache_params = {}
    count = 0 
    p_node = trial.suggest_categorical('p_node', list(np.arange(0.1, 0.9,0.1)))
    p_out = trial.suggest_categorical('p_edge', list(np.arange(0.1, 0.3, 0.1)))
    p_edge = min(0.1, 1.0 - p_node - p_out)
    n_nodes = trial.suggest_categorical('n_nodes', [20, 30, 50, 100])
    return test_evo(n_nodes, p_node, p_edge, 1000, 2000)

if __name__ == "__main__":

    if len(sys.argv) < 3 or sys.argv[2] != "optuna":
        # test_evo(20, 0.5, 0.3, 100000, 200000) # melanoma
        test_evo(30, 0.5, 0.3, 100000, 200000) # Dent
    else:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        print(study.best_params)
