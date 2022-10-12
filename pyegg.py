"""
Equality saturation in Python 3.10+
"""

# Author: Fabricio Olivetti de Franca <folivetti@ufabc.edu.br>
#
# License: MIT

from dataclasses import dataclass
from typing import Any
from db import *
from expr import *

__all__ = [
    "Egg",
]

#print(showTree(Add(Var(0),Const(2.1))))





## SATURATION

def eqSat(scheduler, expr, rules, costFun):
    egraph = Egraph(expr)
    egraph = runEqSat(egraph, scheduler, rules)
    return getBest(cost, egraph)


def runEqSat(egraph, scheduler, rules):
    rule_sched = {}
    for i in range(30):
        old_map_class = egraph.map_class
        old_hashcon = egraph.hashcon
        db = DataBase(egraph)

        # step 1: match the rules
        matches = [] #flatten [match for match in matchWithScheduler(db, i, rule_sched, j, rule) for (j, rule) in enumerate(rules)]

        # step 2: apply matches
        for match in matches:
            applyMatch(egraph, match)  # mutable egraph
            egraph.rebuild()

        new_map_class = egraph.map_class
        new_hashcon = egraph.hashcon
        if (new_map_class == old_map_class) and (new_hashcon == old_hash_con):
            return egraph
    return egraph

def matchWithScheduler(db, i, rule_sched, j, rule):
    if j in rule_sched and rule_sched[j] <= i:
        return []
    matches = ematch(db, rule.source)
    rule_sched[j] = i + 5  # updateStats schd i rw_id rule_sched[rw_id] stats matches
    return [(rule, match) for match in matches]

def isValidConditions(rule, match, egraph):
    return True

def applyMatch(egraph, rule, match):
    if isValidConditions(rule, match, egraph):
        new_eclass = reprPat(egraph, match[1], rule.target)
        egraph.merge(match[1], new_eclass)

def reprPat(egraph, subst_map, target):
    def traverse_target(t):
        if isinstance(t, str):
            t = hashstr(t)
            if t not in subst_map:
                print("ERROR: NO SUBSTUTION FOR ", t)
                exit()
            return subst_map[t]
        match t:
            case Add(l, r):
                el = traverse_target(l)
                er = traverse_target(r)
                e = egraph.add(Add(el, er))
                return e
            case Mul(l, r):
                el = traverse_target(l)
                er = traverse_target(r)
                e = egraph.add(Mul(el, er))
                return e
            case Var(x):
                return egraph.add(Var(x))
            case Const(x):
                return egraph.add(Const(x))
            case _ as unreachable:
                assert_never(unreachable)
    return traverse_target(target)








# Example
pattern = Pattern(Mul("x", Add("y", "z")), Add(Mul("x", "y"), Mul("x", "z")))

eg = EGraph()
expr_to_egraph(Mul(Var(2), Add(Const(2), Var(1))), eg)
print(eg.union_find, eg.map_class, eg.hashcon)
DataBase(eg)
q, r = compileToQuery(pattern.source)
print(q.vars)
for a in q.atoms:
    print(a.classIdOrVar, a.relation)