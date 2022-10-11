"""
Equality saturation in Python 3.10+
"""

# Author: Fabricio Olivetti de Franca <folivetti@ufabc.edu.br>
#
# License: MIT

from dataclasses import dataclass
from typing import Any

__all__ = [
    "DataBase", "Query", "Atom"
]

## DATABASE

class DataBase:
    def __init__(self, egraph):
        self.database = {}  # operator -> IntTrie
        for enode, eclass_id in egraph.hashcon.items():
            self.addToDB(enode, eclass_id)

    def populate(self, trie, ids):
        if len(ids) == 0:
            return None
        if trie is None:
            trie = IntTrie(ids[0])
            tmp = trie
            x_prev = ids[0]
            for x in ids[1:]:
                tmp.trie[x_prev] = IntTrie(x)
                tmp = tmp.trie[x_prev]
                x_prev = x
            return trie
        else:
            trie.keys += [ids[0]]
            next_trie = trie.trie.get(ids[1], None)
            trie.trie[ids[1]] = populate(next_trie, ids[1:])

    def addToDB(self, enode, eclass_id):
        enode = operator(enode)
        ids = [eclass_id] + children(enode)
        trie = self.database.get(enode, None)
        self.database[enode] = self.populate(trie, ids)
            

class IntTrie:
    def __init__(self, x):
        self.keys = set([x])
        self.trie = {}  # {int -> IntTrie()}


## MATCHING-QUERY
class Query:
    def __init__(self, myVars, atoms):
        self.vars = myVars
        self.atoms = atoms  # if this is empty, it is a SelectAllQuery

@dataclass(unsafe_hash=True)
class ClassId:
    cid: int

@dataclass(unsafe_hash=True)
class VarId:
    vid: int

ClassOrVar = ClassId | VarID

class Atom:
    def __init__(self, classIdOrVar : ClassOrVar, t : Expr):
        self.classIdOrVar = classIdOrVar 
        self.relation = t  # tree of classIdOrVar

def compileToQuery(pat):
    if isinstance(pat, str):
        return Query([hash(pat)],[]), pat

    v = 0

    def aux(pat):
       nonlocal v
       if isinstance(pat, str):
           return (hash(pat), [])
       rt = v
       v = v + 1
       match pat:
           case Add(l, r):
               (rl, al) = aux(l)
               (rr, ar) = aux(r)
               atoms = [Atom(rt, Add(rl, rr), True)] + al + ar
           case Mul(l, r):
               (rl, al) = aux(l)
               (rr, ar) = aux(r)
               atoms = [Atom(rt, Mul(rl, rr), True)] + al + ar
           case Const(x):
               atoms = [Atom(rt, Const(x), True)]
           case Var(x):
               atoms = [Atom(rt, Var(x), True)]
           case _ as unreachable:
               print("UNREACH!!!!!!!!!!!!!!!!!", pat)
               return (rt,[])
       return (rt, atoms)

    (root, atoms) = aux(pat) 
    return Query(([root] + unique(variables(pat))), atoms), root

def unique(x):
    return list(set(x))