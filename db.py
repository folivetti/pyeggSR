"""
Equality saturation in Python 3.10+
"""

# Author: Fabricio Olivetti de Franca <folivetti@ufabc.edu.br>
#
# License: MIT

from dataclasses import dataclass
from typing import Any

from expr import *
from egraph import *

## RULES

class Pattern:
    def __init__(self, source, target):
        self.vars = set(self.getvars(source)).union(self.getvars(target))
        self.source = source
        self.target = target
        self.properties = {}  # maps a var to a tuple of properties (is_not_value, is_value, is_negative, is_not_negative, is_zero,...)

    def match(self, egraph):
        for enode, eclass_id in egraph.hashcon.items():
            pass

    def getvars(self, t):
        vars = []
        match t:
            case Add(l, r) | Mul(l, r):
                if isinstance(l, str):
                    vars.append(l)
                else:
                    vars += self.getvars(l)
                if isinstance(r, str):
                    vars.append(r)
                else:
                    vars += self.getvars(r)
                return vars
            case Var(x):
                return []
            case Const(x):
                return []
            case _ as unreachable:
                assert_never(unreachable)

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
        self.keys = {x}
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

ClassOrVar = ClassId | VarId

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
           return (VarId(hash(pat)), [])
       rt = VarId(v)
       v = v + 1
       match pat:
           case Add(l, r):
               (rl, al) = aux(l)
               (rr, ar) = aux(r)
               atoms = [Atom(rt, Add(rl, rr))] + al + ar
           case Mul(l, r):
               (rl, al) = aux(l)
               (rr, ar) = aux(r)
               atoms = [Atom(rt, Mul(rl, rr))] + al + ar
           case Const(x):
               atoms = [Atom(rt, Const(x))]
           case Var(x):
               atoms = [Atom(rt, Var(x))]
           case _ as unreachable:
               print("UNREACH!!!!!!!!!!!!!!!!!", pat)
               return (rt,[])
       return (rt, atoms)

    (root, atoms) = aux(pat) 
    return Query(([root] + unique(getVariables(pat))), atoms), root

def getVariables(pat):
   if isinstance(pat, str):
       return [VarId(hash(pat))]
   match pat:
       case Add(l, r):
           ls = getVariables(l)
           rs = getVariables(r)
           return (ls + rs)
       case Mul(l, r):
           ls = getVariables(l)
           rs = getVariables(r)
           return (ls + rs)
       case Var(x):
           return []
       case Const(x):
           return []

def unique(x):
    return list(set(x))

def genericJoin(db, q):
    variables = orderedVars(q)
    substs = [{}]

    for var in variables:
        substs = [{var:x} | subst 
                    for subst in substs 
                    for x in domainX(var, updateAll(var, x, subst, atoms))
                  ]

def domainX(var, atoms):

    def elemOfAtom(atom):
        if atom.classIdOrVar == var:
            return True
        match atom.t:
          case Add(l, r) | Mul(l, r):
            if l==var or r==var:
              return True
          case _:
            return False
        return False

    filtered = filter(elemOfAtom, atoms)
    return [x for x in intersectAtoms(var, db, filtered)]

def intersectAtoms(var, db, atoms):
    classIds = {}
    for atom in atoms:
        op = operator(atom.t)
        if op in db.database:
            classIds = classIds | intersectTrie(var, {}, db.database[op], [atom.classIdOrVar] + toList(atom.t))
    return classIds

def intersectTrie(var, xs, trie, ids):
    i = ids.pop(0)

    match i:
        case ClassId(x):
            if x in trie.trie:
                return intersectTrie(var, xs, trie.trie[x], ids)
            else:
                return {}
        case VarId(x):
             if x in xs:
                 if xs[x] in trie.trie:
                     return intersectTrie(var, xs, trie.trie[xs[x]], ids)
                 else:
                     return {}
             else:
                 if x == var:
                     def isDiffFrom(y):
                         match y:
                             case ClassId(x):
                                 return False
                             case VarId(x):
                                 return x != y
                     if all(isDiffFrom, ids):
                         return trie.keys
                     else:
                         domains = {}
                         for k, v in trie.trie.items():
                             xs[x] = k
                             if len(intersectTrie(var, xs, v, ids)) > 0:
                                 domains.add(k)
                         return domains
                 else:
                     domains = {}
                     for k, v in trie.trie.items():
                         xs[x] = k
                         domains = domains | intersectTrie(var, xs, v, ids)
                     return domains

def updateAll(var, x, subst, atoms):
    new_atoms = update(var, x, atoms)
    for k,v in subst.items():
        new_atoms = update(k,v,new_atoms)
    return new_atoms

def update(var, x, atoms):
    def replace(old_atom):
        atom = Atom(old_atom.classIdOrVar, old_atom.t, old_atom.isVar)
        if atom.classIdOrVar == var and atom.isVar:
            atom.classIdOrVar = x
            atom.isVar = False
        match atom.t:
           case Mul(l, r):
             newl = x if l==var else l
             newr = x if r==var else r
             atom.t = Mul(newl, newr)
           case Add(l, r):
             newl = x if l==var else l
             newr = x if r==var else r
             atom.t = Add(newl, newr)
           case _ as unmatched: 
             atom.t = unmatched
        return atom
    new_atoms = [replace(atom) for atom in atoms]

def orderedVars(q):
    return q.vars