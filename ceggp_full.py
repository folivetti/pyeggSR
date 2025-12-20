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

def fitness(eg, eclass, params, image, target):
    result, _ = evaluate_egraph(eclass, eg, params, image, False)
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
    # choice an operator from Expr
    for h in range(1, size + 1):
        # list of children are those children from the the previous level
        # children = eids[h-1] 
        # this next line allows children from any previous level
        children = [c for k, v in eids.items() for c in v]
        op = np.random.choice(operators)
        field_values = {}
        for f in fields(op):
            field_values[f.name] = np.random.choice(children)
        eid = egraph.add(op(**field_values))
        eids[h] = [eid]
    return egraph, eids

def get_enode(egraph, eid):
    return next(iter(egraph.map_class[eid].enodes))

def mutate_node(egraph, eids):
    terminals, operators = get_terminals_and_operators(Expr)
    orig_eids = {k: v.copy() for k, v in eids.items()}
    not_new = True 

    while not_new:
        # choose a random point
        column = np.random.choice(list(eids.keys())[1:])
        # choose a random eid from that depth
        eid = np.random.choice(eids[column])
        enode = get_enode(egraph, eid)
        cs = children(enode)
        csnew = [c for k, v in eids.items() for c in v]
        cs += [np.random.choice(csnew), np.random.choice(csnew)]  # add some new children
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

        not_new = False # not evidence
        if not_new:
            # copy back the original eids 
            eids = {k: v.copy() for k, v in orig_eids.items()}
    return egraph, eids, column

def mutate_edge(egraph, eids):
    terminals, operators = get_terminals_and_operators(Expr)
    orig_eids = {k: v.copy() for k, v in eids.items()}
    not_new = True 

    while not_new:
        # choose a random column
        column = np.random.choice(list(eids.keys())[1:])
        # choose a random eid from that column
        eid = np.random.choice(eids[column])
        enode = get_enode(egraph, eid)
        cs = children(enode)
        csnew = [c for k, v in eids.items() for c in v]
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

        not_new = False #not evidence
        if not_new:
            # copy back the original eids 
            eids = {k: v.copy() for k, v in orig_eids.items()}
    return egraph, eids, column
    
def get_n_params(egraph, root):
    enode = next(iter(egraph.map_class[egraph.find(root)].enodes))
    c = number_of_consts(enode)
    for child in children(enode):
        c += get_n_params(egraph, child)
    return c

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

def calc_test_score(egraph, imgs, masks):
    # root is the key of the greatest value in cache_fitness 
    score = 0.0
    for i in range(len(imgs)):
        root = max(cache_fitness, key=lambda k: cache_fitness.get(k))
        params = cache_params[root]
        score += fitness(egraph, root, params, imgs[i], masks[i])
    return score / len(imgs)

def step(egraph, eids, imgs, masks, best_score, it):
    global count
    orig_eids = {k: v.copy() for k, v in eids.items()}
    if random.random() < 0.5:
        egraph, eids, d = mutate_node(egraph, eids)
    else:
        egraph, eids, d = mutate_edge(egraph, eids)
    roots = [v[0] for k,v in eids.items() if len(v)>0 and k <= d]
    new_best = False
    for root in roots:
        score = calc_score(egraph, root, imgs, masks)
        if score >= best_score:
            if score > best_score:
                print(f"New best score: {score} at iteration {it} with {count} evaluations")
                print(">>", root, showEGraph(egraph, root))
            if score > best_score or egraph.map_class[egraph.find(root)].height <= egraph.map_class[egraph.find(max(cache_fitness, key=cache_fitness.get))].height:
                best_score = score
                new_best = True
    if not new_best:
        return egraph, orig_eids, best_score
    return egraph, eids, best_score

def load_datasets(dirname):
    df = pd.read_csv(f"datasets/{dirname}/dataset.csv", sep=";")
    imgs = [_read_inputs(f"datasets/{dirname}/{v}")[0] for v in df[df.set == "training"].input.to_list()]
    masks = [_read_mask(f"datasets/{dirname}/{v}", imgs[i][0].shape) for i,v in enumerate(df[df.set == "training"]["label"].to_list())]

    test_imgs = [_read_inputs(f"datasets/{dirname}/{v}")[0] for v in df[df.set == "testing"].input.to_list()]
    test_masks = [_read_mask(f"datasets/{dirname}/{v}", test_imgs[i][0].shape) for i,v in enumerate(df[df.set == "testing"]["label"].to_list())]

    return imgs, masks, test_imgs, test_masks

def test_evo():
    global count
    imgs, masks, test_imgs, test_masks = load_datasets("melanoma")

    best_score = 0.0
    egraph, eids = get_fst_rnd_egraph(50)
    roots = [v[0] for k,v in eids.items() if len(v)>0]
    for root in roots:
        score = calc_score(egraph, root, imgs, masks)
        if score >= best_score:
            print(f"New best score: {score}")
            print(">>", root, showEGraph(egraph, root))
            best_score = score

    # for iteration in range(1000):
    iteration = 0 
    while count < 5000 and iteration < 5550:
        egraph, eids, best_score = step(egraph, eids, imgs, masks, best_score, iteration)
        iteration += 1
    test_score = calc_test_score(egraph, test_imgs, test_masks)
    print(f"Final test score: {test_score} with {count} evaluations")
    #for i in range(len(test_imgs)):
    #    root = max(cache_fitness, key=lambda k: cache_fitness.get(k)[i])
    #    print(">>", root, showEGraph(egraph, root))
    # print(">>", max(cache_fitness, key=cache_fitness.get), showEGraph(egraph, max(cache_fitness, key=cache_fitness.get)))


def test_opt():
    img, sz = _read_inputs("datasets/MelanLiza/images/NVA_19-002.MelanLizaV10_2.S1528447.P4595.png")
    mask = _read_mask("datasets/MelanLiza/masks/NVA_19-002.MelanLizaV10_2.S1528447.P4595.zip", img[0].shape)
    '''
    egraph = EGraph()
    id0 = egraph.add(Var(0))
    id1 = egraph.add(Var(1))
    id2 = egraph.add(Var(2))
    id3 = egraph.add(Sub(id0, id1))
    id4 = egraph.add(BinaryInRange(id2)) # 2
    id5 = egraph.add(GaussianBlur(id3)) # 1
    id6 = egraph.add(Laplacian(id5))
    id7 = egraph.add(MorphBlackHat(id6)) # 2
    id8 = egraph.add(Min(id7,id4))
    id9 = egraph.add(FillHoles(id8))
    id10 = egraph.add(MedianBlur(id9)) # 1
    '''
    egraph, eids = get_fst_rnd_egraph(5)
    root = eids[5][0]
    print(">>", showEGraph(egraph, root))

    n_params = get_n_params(egraph, root)
    print(f"Number of params: {n_params}")
    params = np.random.randint(0, 255, size=(n_params,)).tolist()

    result, _ = evaluate_egraph(root, egraph, params, img)
    #cv2.imwrite("test_output.png", result)
    score = fitness(egraph, root, params, img, mask)
    max_x, max_score = -1, score
    #for _ in range(10):
    for i, x in enumerate(params):
        break
        max_x = x
        for x in range(256):
            params[i] = x
            score = fitness(egraph, root, params, img, mask)
            if score > max_score:
                max_x, max_score = x, score
                print("New score: ", max_score, i, x)
        params[i] = max_x

    print(f"Params: {params}, IOU: {max_score}")

if __name__ == "__main__":
    
    try:
        test_evo()
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
