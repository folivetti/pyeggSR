"""
Equality saturation in Python 3.10+
"""

# Author: Fabricio Olivetti de Franca <folivetti@ufabc.edu.br>
#
# License: MIT

from dataclasses import dataclass
from typing import Any
from expr import *

class EGraph:

    def __init__(self):
        self.union_find = {} # maps an eclass_id to another eclass_id
        self.map_class = {} # maps an eclass_id to the eclass object
        self.hashcon = {} # maps an enode to its eclass_id
        self.worklist = [] # enodes and eclass_ids that needs to be revisited
        self.analysis = []

    def canonicalize(self, enode):
        '''
        An enode is a node where its children points to
        eclasses. Since an eclass may have multiple labels,
        this function replaces the current ids with the canonical ids.
        '''
        return applyTree(self.find, enode)

    def find(self, eclass_id):
        if self.union_find[eclass_id] == eclass_id:
            return eclass_id
        return self.find(self.union_find[eclass_id])

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

        old_leader_data, old_leader_parents = leader_class.eclass_data, leader_class.parents
        old_sub_data, old_sub_parents = sub_class.eclass_data, sub_class.parents
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

def expr_to_egraph(expr, egraph):
    new_children = []
    for child in children(expr):
        new_children.append(expr_to_egraph(child, egraph))
    n = replaceChildren(expr, new_children)
    return egraph.add(n)