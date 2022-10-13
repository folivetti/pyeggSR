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

def hashstr(x: str):
    acc = 5381
    for c in x:
        acc = 33*acc ^ ord(c)
    return acc

## RULES

class Pattern:
    def __init__(self, source, target):
        self.vars = set(self.getvars(source)).union(self.getvars(target))
        self.source = source
        self.target = target
        self.properties = {}  # maps a var to a tuple of properties (is_not_value, is_value, is_negative, is_not_negative, is_zero,...)

    # TODO: incorporate match into pattern
    def match(self, egraph):
        for enode, eclass_id in egraph.hashcon.items():
            pass

    def getvars(self, t):
        if isinstance(t, str):
            return [t]
        return [v for c in children(t) for v in self.getvars(c)]

## DATABASE

class DataBase:
    def __init__(self, egraph):
        self.database = {}  # operator -> IntTrie
        for enode, eclass_id in egraph.hashcon.items():
            self.addToDB(enode, eclass_id)

    def populate(self, trie, ids):
        if len(ids) == 0:
            return IntTrie(None) if trie is None else trie
        if trie is None:
            trie = IntTrie(ids[0])
            trie.trie[ids[0]] = self.populate(None, ids[1:])
            return trie
        else:
            trie.keys |= {ids[0]}
            next_trie = trie.trie.get(ids[1], None)
            trie.trie[ids[0]] = self.populate(next_trie, ids[1:])
            return trie

    def addToDB(self, enode, eclass_id):
        ids = [eclass_id] + children(enode)
        enode = operator(enode)
        trie = self.database.get(enode, None)
        self.database[enode] = self.populate(trie, ids)
            

class IntTrie:
    def __init__(self, x):
        self.keys = {x}
        self.trie = {}  # {int -> IntTrie()}

## SATURATION
def eqSat(scheduler, expr, rules, costFun):
    egraph = EGraph()
    eclassId = expr_to_egraph(expr, egraph)
    runEqSat(egraph, scheduler, rules)
    return getBest(costFun, egraph, eclassId)[1]

def getBest(costFun, egraph, eclassId):
    start_eclass = egraph.find(eclassId)
    cost = None
    expr = None
    for n in egraph.map_class[start_eclass].enodes:
        c = costFun(n)
        new_children = []
        for child in children(n):
            (cn, nn) = getBest(costFun, egraph, child)
            c = c + cn
            new_children.append(nn)
        if cost is None or c < cost:
            cost = c
            expr = replaceChildren(n, new_children)
    return (cost, expr)
    
def runEqSat(egraph, scheduler, rules):
    rule_sched = {}
    for i in range(30):
        old_map_class = egraph.map_class
        old_hashcon = egraph.hashcon
        db = DataBase(egraph)

        # step 1: match the rules
        matches = [] 
        for j, r in enumerate(rules):
            m = matchWithScheduler(db, i, rule_sched, j, r) # check if rule_sched is mutable
            matches += m

        #flatten [match for match in matchWithScheduler(db, i, rule_sched, j, rule) for (j, rule) in enumerate(rules)]

        # step 2: apply matches
        for match in matches:
            applyMatch(egraph, *match)  # mutable egraph
        egraph.rebuild()

        new_map_class = egraph.map_class
        new_hashcon = egraph.hashcon
        if (new_map_class == old_map_class) and (new_hashcon == old_hashcon):
            return egraph
    return egraph

def matchWithScheduler(db, i, rule_sched, j, rule):
    if j in rule_sched and rule_sched[j] <= i:
        return []
    matches = ematch(db, rule.source)
    rule_sched[j] = i + 5  # 5 iterations ban -- updateStats schd i rw_id rule_sched[rw_id] stats matches
    return [(rule, match) for match in matches]

def isValidConditions(rule, match, egraph):
    return True # TODO: check the preconditions of the rule

def applyMatch(egraph, rule, match):
    if isValidConditions(rule, match, egraph):
        new_eclass = reprPat(egraph, match[0], rule.target)
        egraph.merge(match[1].cid, new_eclass)

def reprPat(egraph, subst_map, target):
    def traverse_target(t):
        if isinstance(t, str):
            t = VarId(hashstr(t))
            if t not in subst_map:
                print("ERROR: NO SUBSTUTION FOR ", t, subst_map)
                exit()
            return subst_map[t].cid
        new_children = [traverse_target(c) for c in children(t)]
        return egraph.add(replaceChildren(t, new_children))

    return traverse_target(target)

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

def ematch(db, source):
    (q, root) = compileToQuery(source)
    substs = genericJoin(db, q)
    matches = []
    for s in substs:
        if len(s) > 0:
            if root in s:
                matches.append((s, s[root]))
            else:
                print("ROOT NOT IN S!") 
    return matches

def compileToQuery(pat):
    if isinstance(pat, str):
        return Query([hashstr(pat)],[]), pat

    v = 0

    def aux(pat):
       nonlocal v
       if isinstance(pat, str):
           return (VarId(hashstr(pat)), [])
       rt = VarId(v)
       v = v + 1
       atoms = []
       new_children = []
       for child in children(pat):
          rc, ac = aux(child)
          new_children.append(rc)
          atoms += ac
       atoms = [Atom(rt, replaceChildren(pat, new_children))] + atoms
       return (rt, atoms)

    (root, atoms) = aux(pat) 
    return Query(([root] + unique(getVariables(pat))), atoms), root

def getVariables(pat):
   if isinstance(pat, str):
       return [VarId(hashstr(pat))]
   return [var for child in children(pat) for var in getVariables(child)]
   
def unique(x):
    return list(set(x))

def printAtoms(atoms):
    for a in atoms:
        print("a ", a.classIdOrVar, " <=> ", showTree(a.relation))

def genericJoin(db, q):

    def genericrec(atoms, variables):
        substs = []
        if len(variables) == 0:
           return [{} for _ in atoms]
        x = variables.pop(0)
        for classId in domainX(db, x, atoms):
            newatoms = updateAll(x, classId, atoms)
            for y in genericrec(newatoms, variables.copy()):
                y[x] = classId
                substs.append(y)
        return substs

    return genericrec(q.atoms, orderedVars(q))

def domainX(db, var, atoms):

    def elemOfAtom(atom):
        match atom.classIdOrVar:
            case VarId(x):
                return atom.classIdOrVar == var
            case _:
                return any(x==var for x in children(atom.relation))
        return False

    filtered = list(filter(elemOfAtom, atoms))
    return [ClassId(x) for x in intersectAtoms(var, db, filtered)]

def intersectAtoms(var, db, atoms):
    if len(atoms) == 0:
        return set([])
    def f(atom):
        op = operator(atom.relation)
        if op in db.database:
            moreIds = intersectTrie(var, {}, db.database[op], [atom.classIdOrVar] + children(atom.relation))
            if moreIds is None:
                print("ERROR!! TRIE NONE!!!\n\n\n\n")
                return None
            else:
                return moreIds
        else:
            return set([])

    classIds = f(atoms[0])
    for atom in atoms[1:]:
        classIds = classIds & f(atom)
    return classIds

def intersectTrie(var, xs, trie, ids):
    if len(ids) == 0:
        return set([])

    i = ids.pop(0)

    match i:
        case ClassId(x):
            if x in trie.trie:
                return intersectTrie(var, xs, trie.trie[x], ids)
            else:
                return None
        case VarId(x):
             if x in xs:
                 if xs[x] in trie.trie:
                     return intersectTrie(var, xs, trie.trie[xs[x]], ids)
                 else:
                     return None
             else:
                 if VarId(x) == var:
                     def isDiffFrom(y):
                         nonlocal x
                         match y:
                             case ClassId(z):
                                 return False
                             case VarId(z):
                                 return VarId(x) != y
                     if all(map(isDiffFrom, ids)) or len(ids)==0:
                         return trie.keys
                     else:
                         domains = set([])
                         for k, v in trie.trie.items():
                             xs[x] = k
                             if intersectTrie(var, xs, v, ids) is not None:
                                 domains.add(k)
                         return domains
                 else:
                     domains = set([])
                     for k, v in trie.trie.items():
                         xs[x] = k
                         t = intersectTrie(var, xs, v, ids)
                         if t is not None:
                             domains = domains | t
                     return domains
        case x as unmatched:
            print("WARNING!WARNING!", x)

def updateAll(var, classId, atoms):
    return update(var, classId, atoms)

def update(var, x, atoms):
    def replace(old_atom):
        atom = Atom(old_atom.classIdOrVar, old_atom.relation)
        if atom.classIdOrVar == var:
            atom.classIdOrVar = x
        new_children = [x if c==var else c for c in children(atom.relation)]
        atom.relation = replaceChildren(atom.relation, new_children)
        return atom
    return [replace(atom) for atom in atoms]

def orderedVars(q):
    def elemOfAtom(v, atom):
        match atom.classIdOrVar:
            case VarId(x):
                return atom.classIdOrVar == v
            case _:
                return any(x==v for x in children(atom.relation))
        return False

    def varcost(var):
        acc = 0
        for atom in q.atoms:
            if elemOfAtom(var, atom):
                acc = acc - 100 + atomLen(atom)
        return acc

    def atomLen(atom):
        return 1 + len(children(atom.relation))
            
    variables = []
    for atom in q.atoms:
        for a in [atom.classIdOrVar] + children(atom.relation):
            if a not in variables:
                match a:
                    case VarId(x):
                        variables.append(a)
                    case _:
                        pass
    return sorted(variables, key=varcost)