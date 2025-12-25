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
import pstats
from pstats import SortKey
import optuna 

DIM = 6  # number of input channels
cache_fitness = {}
cache_params = {}
count = 0 

def fitness(eg, eclass, params, image, target, useCache=False, isTest=False):
    result, _ = evaluate_egraph(eclass, eg, params, image, useCache)
    iou = compute_iou(target, result)
    return iou

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
        children = [c for k, v in eids.items() for c in v]
        op = np.random.choice(operators)
        field_values = {}
        for f in fields(op):
            field_values[f.name] = np.random.choice(children)
        eid = egraph.add(op(**field_values))
        eids[h] = [eid]
    out = np.random.choice(size) + 1
    return egraph, eids, out 

def get_enode(egraph, eid):
    return next(iter(egraph.map_class[eid].enodes))

def mutate_node(egraph, eids, out):
    terminals, operators = get_terminals_and_operators(Expr)
    orig_eids = {k: v.copy() for k, v in eids.items()}
    not_new = True 
    attempt = 0

    while not_new and attempt < 5:
        # choose a random point
        column = np.random.choice(list(eids.keys())[1:])
        # choose a random eid from that depth
        eid = np.random.choice(eids[column])
        enode = get_enode(egraph, eid)
        cs = children(enode)
        csnew = [c for k, v in list(eids.items())[:column] for c in v]
        cs += [np.random.choice(csnew), np.random.choice(csnew)]  # add some new children in case the new node requires more
        # choose a random operator
        op = np.random.choice(operators)
        field_values = {}

        for f, c in zip(fields(op), cs):
            field_values[f.name] = c 
        new_op = op(**field_values)
        evidence = egraph.hashcon.get(new_op, None) is None
        new_eid = egraph.add(new_op)
        eids[column] = [new_eid]

        # update the other eids dict entries if needed
        for d in range(column + 1, max(eids.keys()) + 1):
            enode = get_enode(egraph, eids[d][0])
            cs = children(enode)
            # replace any occurrence of eid with new_eid
            for i in range(len(cs)):
                if cs[i] == eid:
                    cs[i] = new_eid
            n = replaceChildren(enode, cs)
            # check if n is new or not
            evidence = evidence and (egraph.hashcon.get(n, None) is None)
            updated_eid = egraph.add(n)
            eids[d] = [updated_eid]

        not_new = False # not evidence
        attempt += 1
        if not_new:
            # copy back the original eids 
            eids = {k: v.copy() for k, v in orig_eids.items()}
    return egraph, eids, column, out

def mutate_edge(egraph, eids, out):
    terminals, operators = get_terminals_and_operators(Expr)
    orig_eids = {k: v.copy() for k, v in eids.items()}
    not_new = True 
    attempt = 0

    while not_new and attempt < 5:
        # choose a random column
        column = np.random.choice(list(eids.keys())[1:])
        # choose a random eid from that column
        eid = np.random.choice(eids[column])
        enode = get_enode(egraph, eid)
        cs = children(enode)
        csnew = [c for k, v in list(eids.items())[:column] for c in v]
        ix = np.random.randint(0, len(cs))
        cs[ix] = np.random.choice(csnew)
        # choose a random operator
        new_op = replaceChildren(enode, cs)

        evidence = egraph.hashcon.get(new_op, None) is None
        new_eid = egraph.add(new_op)
        eids[column] = [new_eid]

        # update the other eids dict entries if needed
        for d in range(column + 1, max(eids.keys()) + 1):
            enode = get_enode(egraph, eids[d][0])
            cs = children(enode)
            # replace any occurrence of eid with new_eid
            for i in range(len(cs)):
                if cs[i] == eid:
                    cs[i] = new_eid
            n = replaceChildren(enode, cs)
            # check if n is new or not
            evidence = evidence and (egraph.hashcon.get(n, None) is None)
            updated_eid = egraph.add(n)
            eids[d] = [updated_eid]

        not_new = False # not evidence
        attempt += 1
        if not_new:
            # copy back the original eids 
            eids = {k: v.copy() for k, v in orig_eids.items()}
    return egraph, eids, column, out
    
def mutate_out(egraph, eids, out):
    out = np.random.choice(len(eids))
    return egraph, eids, out

def get_n_params(egraph, root):
    enode = next(iter(egraph.map_class[egraph.find(root)].enodes))
    c = number_of_consts(enode)
    for child in children(enode):
        c += get_n_params(egraph, child)
    return c

def calc_score(egraph, root, imgs, masks, recalc=False):
    global count

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
    return score

def calc_test_score(egraph, imgs, masks):
    # root is the key of the greatest value in cache_fitness 
    score = 0.0
    root = max(cache_fitness, key=lambda k: cache_fitness.get(k))
    params = cache_params[root]
    for i in range(len(imgs)):
        score += fitness(egraph, root, params, imgs[i], masks[i], False)
    return score / len(imgs)

def step(egraph, eids, out, imgs, masks, best_score, it, p_node = 0.25, p_edge = 0.5, p_accept = 0.1):
    global count
    orig_eids = {k: v.copy() for k, v in eids.items()}
    orig_out = out
    r = random.random()
    recalc = False 
    if r < p_node:
        egraph, eids, d, out = mutate_node(egraph, eids, out)
    elif r < p_node + p_edge:
        egraph, eids, d, out = mutate_edge(egraph, eids, out)
    else:
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
    #if not new_best and np.random.random() >= p_accept:
    #    return egraph, orig_eids, best_score, orig_out
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
    imgs, masks, test_imgs, test_masks = load_datasets("melanoma")

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
    return test_evo(n_nodes, p_node, p_edge, 10000, 10000)

if __name__ == "__main__":

    test_evo(50, 0.6, 0.3, 100000, 600000)
    #study = optuna.create_study(direction='maximize')
    #study.optimize(objective, n_trials=100)
    #print(study.best_params)
