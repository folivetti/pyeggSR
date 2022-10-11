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

class EGraph:

    def __init__(self):
        self.union_find = {}
        self.map_class = {}
        self.hashcon = {}
        self.worklist = []
        self.analysis = []

    def canonicalize(self, enode):
        return applyTree(self.find, enode)
        match enode:
            case Add(l, r):
                return Add(self.find(l), self.find(r))
            case Mul(l, r):
                return Mul(self.find(l), self.find(r))
            case Const(x):
                return Const(x)
            case Var(x):
                return Var(x)

    def add(self, enode):
        enode = self.canonicalize(enode)
        if enode in self.hashcon:
            return self.hashcon[enode]
        else:
            eclass_id = len(self.map_class)
            self.map_class[eclass_id] = EClass(eclass_id, enode)
            self.union_find[eclass_id] = eclass_id
            for eclass in children(enode):
                self.map_class[eclass].parents.append((eclass_id, enode))
            self.hashcon[enode] = eclass_id
            self.worklist.append((enode, eclass_id))
            # TODO: modifyA eclass_id self
            return eclass_id

    def merge(self, eclass_id_1, eclass_id_2):
        if self.find(eclass_id_1) == self.find(eclass_id_2):
            return self.find(eclass_id_1)
        # must put something on worklist. Check paper
        eclass_1 = self.map_class[eclass_id_1]
        eclass_2 = self.map_class[eclass_id_2]
        if len(eclass_1.parents) > len(eclass_2.parents):
            leader, leader_class = eclass_id_1, eclass_1
            sub, sub_class = eclass_id_2, eclass_2
        else:
            leader, leader_class = eclass_id_2, eclass_2
            sub, sub_class = eclass_id_1, eclass_1
        self.union_find[sub] = leader
        old_leader_data, old_leader_parents = leader.eclass_data, leader.parents
        old_sub_data, old_sub_parents = sub.eclass_data, sub.parents
        leader_class.parents += sub_class.parents
        leader_class.enodes.union(sub_class.enodes)
        # joinA leader_class.eclass_data sub_class.eclass_data
        leader_class.eclass_data = None
        self.map_class.pop(sub, None)
        self.map_class[leader] = leader_class
        self.worklist = sub_class.parents + self.worklist
        analysis = [] if leader_class.eclass_data == old_leader_data else old_leader_parents
        analysis += [] if sub_class.eclass_data == old_sub_data else old_sub_parents
        self.analysis = analysis + self.analysis
        # TODO: modifyA leader self

        return leader

    def rebuild(self):
        worklist = self.worklist
        analysis = self.analysis
        self.worklist, self.analysis = [], []
        for wl in worklist:
            self.repair(wl)
        for al in analysis:
            self.repair_anaylsis(al)
        if len(self.worklist) != 0 or len(self.analysis) != 0:
            self.rebuild()
        return

    def repair(self, wl):
        enode, eclass_id = wl
        self.hashcon.pop(enode, None)
        enode = self.canonicalize(enode)
        eclass_id = self.find(eclass_id)
        if enode in self.hashcon:
            self.merge(self.hashcon[enode], eclass_id)
        return

    def repair_anaylsis(self, al):
        enode, eclass_id = al
        canon_id = self.find(eclass_id)
        eclass = self.map_class[canon_id]
        new_data = eclass.eclass_data  # joinA (eclass.eclass_data, makeA(enode))
        if eclass.eclass_data != new_data:
            self.analysis = eclass.parents + self.analysis
            self.map_class[canon_id].eclass_data = new_data
            # modifyA(canon_id)

    def find(self, eclass_id):
        if self.union_find[eclass_id] == eclass_id:
            return eclass_id
        return self.find(self.union_find[eclass_id])


class EClass:
    def __init__(self, cid, enode):
        self.id = cid
        self.enodes = set([enode])
        match enode:
            case Const(x):
                self.eclass_data = x
            case _:
                self.eclass_data = None  # this should actually expand to recursive calls, need egraph for this
        self.parents = []


def children(enode):
    match enode:
        case Add(l, r):
            return [l, r]
        case Mul(l, r):
            return [l, r]
        case Var(x):
            return []
        case Const(x):
            return []
        case _ as unreachable:
            assert_never(unreachable)

def operator(enode):
    match enode:
        case Add(l, r):
            return Add(None, None)
        case Mul(l, r):
            return Mul(None, None)
        case Var(x):
            return Var(x)
        case Const(x):
            return Const(x)
        case _ as unreachable:
            assert_never(unreachable)

def expr_to_egraph(expr, egraph):
    match expr:
        case Const(x):
            return egraph.add(expr)
        case Var(x):
            return egraph.add(expr)
        case Add(l, r):
            c_id1 = expr_to_egraph(l, egraph)
            c_id2 = expr_to_egraph(r, egraph)
            return egraph.add(Add(c_id1, c_id2))
        case Mul(l, r):
            c_id1 = expr_to_egraph(l, egraph)
            c_id2 = expr_to_egraph(r, egraph)
            return egraph.add(Mul(c_id1, c_id2))

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

def applyMatch(egraph, rule, match):
    if isValidConditions(rule, match, egraph):
        new_eclass = reprPat(egraph, match.subst_map, rule.target)
        egraph.merge(match.eclass, new_eclass)

def reprPat(egraph, subst_map, target):
    def traverse_target(t):
        if isinstance(t, str):
            if t not in subst_map:
                print("ERROR: NO SUBSTUTION FOR " + t)
                exit()
            return subst_map[t]
        match t:
            case Add(l, r):
                return Add(traverse_target(l), traverse_target(r))
            case Mul(l, r):
                return Mul(traverse_target(l), traverse_target(r))
            case Var(x):
                return Var(x)
            case Const(x):
                return Const(x)
            case _ as unreachable:
                assert_never(unreachable)
    n = traverse_target(target)
    eclass_id = egraph.add(n)
    return eclass_id



def variables(pat):
   if isinstance(pat, str):
       return [hash(pat)]
   match pat:
       case Add(l, r):
           ls = variables(l)
           rs = variables(r)
           return (ls + rs)
       case Mul(l, r):
           ls = variables(l)
           rs = variables(r)
           return (ls + rs)
       case Var(x):
           return []
       case Const(x):
           return []

def ematch(db, source):
    (q, root) = compileToQuery(source)   
    #return mapMaybe f (genericJoin db q)

def genericJoin(db, q):
    variables = orderedVars(q)
    substs = [{}]

    for var in variables:
        substs = [{var:x} | subst 
                    for subst in substs 
                    for x in domainX(var, updateAll(var, x, subst, atoms))
                  ]

def domainX(var, atoms):
    filtered = filter(elemOfAtom, atoms)
    return [x for x in intersectAtoms(var, db, filtered)]

def elemOfAtom(atom):
    if atom.isVar and atom.classIdOrVar == var:
        return True
    match atom.t:
      case Add(l, r) | Mul(l, r):
        if l==var or r==var:
          return True
      case _:
        return False
    return False

def intersectAtoms(var, db, atoms):
    classIds = {}
    for atom in atoms:
        op = operator(atom.t)
        if op in db.database:
            classIds = classIds | intersectTrie(var, {}, db.database[op], [atom.classIdOrVar] + toList(atom.t))
    return classIds

def intersectTrie(var, xs, trie, ids):
    

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