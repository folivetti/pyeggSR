"""
Test suite for the evaluate function.
This demonstrates how to use the evaluate function to evaluate expression trees.
"""

import numpy as np
from expr import *
from egraph import *
from evaluate import *
from readimage import _read_inputs, _read_mask
import cv2 
import random
from dataclasses import dataclass, fields, is_dataclass

DIM = 3  # number of input channels
cache_fitness = {}

def fitness(eg, eclass, params, image, target):
    result, _ = evaluate_egraph(eclass, eg, params, image)
    iou = compute_iou(target, result)
    return iou

def get_fst_rnd_egraph(depth):
    terminals, operators = get_terminals_and_operators(Expr)
    egraph = EGraph()
    # insert all variables up to d
    eids = {}
    eids[0] = []
    for i in range(DIM):
        eid = egraph.add(Var(i))
        eids[0].append(eid)
    # choice an operator from Expr
    for h in range(1, depth + 1):
        # list of children are those children from the the previous level
        children = eids[h-1] 
        # this next line allows children from any previous level
        # children = [eids[hh][i] for hh in range(h) for i in range(len(eids[hh]))]
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
        # choose a random depth
        depth = np.random.choice(list(eids.keys())[1:])
        # choose a random eid from that depth
        eid = np.random.choice(eids[depth])
        enode = get_enode(egraph, eid)
        cs = children(enode)
        csnew = [eids[hh][i] for hh in range(depth) for i in range(len(eids[hh]))]
        cs += [np.random.choice(csnew), np.random.choice(csnew)]  # add some new children
        # choose a random operator
        op = np.random.choice(operators)
        field_values = {}

        for f, c in zip(fields(op), cs):
            field_values[f.name] = c 
        new_op = op(**field_values)
        evidence = egraph.hashcon.get(new_op, None) is None
        new_eid = egraph.add(new_op)
        new_depth = egraph.map_class[new_eid].height
        eids[new_depth] = [new_eid]

        # update the other eids dict entries if needed
        for d in range(new_depth + 1, max(eids.keys()) + 1):
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
    return egraph, eids

def mutate_edge(egraph, eids):
    terminals, operators = get_terminals_and_operators(Expr)
    orig_eids = {k: v.copy() for k, v in eids.items()}
    not_new = True 

    while not_new:
        # choose a random depth
        depth = np.random.choice(list(eids.keys())[1:])
        # choose a random eid from that depth
        eid = np.random.choice(eids[depth])
        enode = get_enode(egraph, eid)
        cs = children(enode)
        csnew = [eids[hh][i] for hh in range(depth) for i in range(len(eids[hh]))]
        cs = [np.random.choice(csnew)] if len(cs) == 1 else [np.random.choice(csnew), np.random.choice(csnew)]  # add some new children
        # choose a random operator
        new_op = replaceChildren(enode, cs)

        evidence = egraph.hashcon.get(new_op, None) is None
        new_eid = egraph.add(new_op)
        new_depth = egraph.map_class[new_eid].height
        eids[new_depth] = [new_eid]

        # update the other eids dict entries if needed
        for d in range(new_depth + 1, max(eids.keys()) + 1):
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
    return egraph, eids
    
def get_n_params(egraph, root):
    enode = next(iter(egraph.map_class[egraph.find(root)].enodes))
    c = number_of_consts(enode)
    for child in children(enode):
        c += get_n_params(egraph, child)
    return c

def calc_score(egraph, root, img, mask):
    if root in cache_fitness:
        return cache_fitness[root]
    n_params = get_n_params(egraph, root)
    params = np.random.randint(0, 255, size=(n_params,)).tolist()
    score = fitness(egraph, root, params, img, mask)
    cache_fitness[root] = score
    return score

def step(egraph, eids, img, mask, best_score):
    orig_eids = {k: v.copy() for k, v in eids.items()}
    if random.random() < 0.5:
        egraph, eids = mutate_node(egraph, eids)
    else:
        egraph, eids = mutate_edge(egraph, eids)
    roots = [v[0] for k,v in eids.items() if len(v)>0]
    new_best = False
    for root in roots:
        score = calc_score(egraph, root, img, mask)
        if score > best_score:
            print(f"New best score: {score}")
            print(">>", root, showEGraph(egraph, root))
            best_score = score
            new_best = True
    if not new_best:
        return egraph, orig_eids, best_score
    return egraph, eids, best_score

def test_evo():
    img, sz = _read_inputs("datasets/MelanLiza/images/NVA_19-002.MelanLizaV10_2.S1528447.P4595.png")
    mask = _read_mask("datasets/MelanLiza/masks/NVA_19-002.MelanLizaV10_2.S1528447.P4595.zip", img[0].shape)

    best_score = 0.0
    egraph, eids = get_fst_rnd_egraph(5)
    roots = [v[0] for k,v in eids.items() if len(v)>0]
    for root in roots:
        score = calc_score(egraph, root, img, mask)
        if score > best_score:
            print(f"New best score: {score}")
            print(">>", root, showEGraph(egraph, root))
            best_score = score

    for iteration in range(100):
        print(f"=== Iteration {iteration} ===")
        egraph, eids, best_score = step(egraph, eids, img, mask, best_score)


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
