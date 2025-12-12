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
        '''
        An e-graph is composed of:

        - union_find: a Union-find structure that maps and e-class id to another e-class id,
                      whenever we add a new e-node that belongs to an existing e-class, 
                      it is initially assigned a new e-class id and merged to the existing e-class.
                      The newly created e-class id points to the original e-class for reference.

        - map_class: is a dictionary that maps an e-class id to the object e-class
        - hashcon: is a dictionary that maps an e-node to its e-class id
        - worklist, analysis: stores the e-nodes and e-classes that must be analysed for repair 
        '''
        self.union_find = {} # maps an eclass_id to another eclass_id
        self.map_class = {} # maps an eclass_id to the eclass object
        self.hashcon = {} # maps an enode to its eclass_id
        self.worklist = [] # enodes and eclass_ids that needs to be revisited
        self.analysis = []
        self.next_id = 0 # next available e-class id

    def canonicalize(self, enode):
        '''
        An e-node is a node where its children (if any) points
        to e-class id. Since every e-node inside an e-class is an
        equivalent expression, this creates multiple paths in the graph
        to write equivalent expressions departing from one e-node.

        Examples of e-node:
            Add(0, 1) - Add any node from e-class 0 to any node of e-class 1
            Log(0) - apply log to any e-node of e-class 0
            Const(1.0) - Just the constant value, 
                          without pointing to any e-class 

        Since during the update of the e-graph, some e-classes may be merged,
        we need to make sure the e-node childs point to the cannonical e-class.

        This is done by applying `find` to each child.
        '''
        return applyTree(self.find, enode)

    def find(self, eclass_id):
        '''
        finds the cannonical e-class id, it just traverse
        the union_find departing from eclass_id until it finds
        the id such that union_find[id] = id.
        '''
        while self.union_find[eclass_id] != eclass_id:
            eclass_id = self.union_find[eclass_id]
        return eclass_id

    def add(self, enode, consts=[]):
        '''
        Add a new e-node:

            1. Make sure the e-class ids of the children are cannonical
            2. If after canonicalizing the e-node, one can find it at our hashcon, return the enode
            3. Otherwise, create a new e-class with this e-node and an unused id 
            4. Make this e-class id canonical by inserting at union_find 
            5. Update the parents of the child e-classes to point to this e-node
            6. Insert the e-node into hashcon
            7. Append to the worklist so we can repair the graph, if needed.
            8. Evaluate e-node, if it returns a float then store it on e-class data (TODO)
        '''
        enode = self.canonicalize(enode)
        if enode in self.hashcon:
            return self.hashcon[enode]
        else:
            eclass_id = self.next_id # len(self.map_class)
            self.next_id += 1

            h = 0 if len(children(enode)) == 0 else max([self.map_class[eclass].height for eclass in children(enode)])

            self.map_class[eclass_id] = EClass(eclass_id, enode, consts, h+1)
            self.union_find[eclass_id] = eclass_id
            for eclass in children(enode):
                self.map_class[eclass].parents.append((eclass_id, enode))
            self.hashcon[enode] = eclass_id
            self.worklist.append((enode, eclass_id))
            # TODO: makeA node egraph
            return eclass_id

    def merge(self, eclass_id_1, eclass_id_2):
        '''
        Merge two equivalent e-classes.

        If the canonical of both e-classes are the same, they are already linked.

        Select the e-class with more parents as the leader, and the other as sub.

        Link the sub to the leader with the union_find, 
        add the sub parents and e-nodes to the leader 
        and remove the sub from map_class. 
        The union_find will still keep the reference 
        to the sub e-class id so we can reach their info.
        '''
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
        leader_class.enodes = leader_class.enodes.union(sub_class.enodes)
        leader_class.height = min(leader_class.height, sub_class.height)
        # joinA leader_class.eclass_data sub_class.eclass_data
        leader_class.eclass_data = None
        self.map_class.pop(sub, None)
        self.map_class[leader] = leader_class

        # store the parents in the worklist for repairing
        self.worklist = sub_class.parents + self.worklist
        # if the data has changed, add the new data for analysis
        analysis = [] if leader_class.eclass_data == old_leader_data else old_leader_parents
        analysis += [] if sub_class.eclass_data == old_sub_data else old_sub_parents
        self.analysis = analysis + self.analysis
        # TODO: modifyA leader self -- prune the e-class so it will keep only the constant value

        return leader

    def rebuild(self):
        '''
        rebuild the e-graph by repairing all:
            e-node, e-class id pairs from the worklist
            parent e-nodes from analysis

        It can happen that after repairing everything, new items are added to
        the lists.
        '''
        while len(self.worklist) != 0 or len(self.analysis) != 0:
            worklist = self.worklist
            analysis = self.analysis
            self.worklist, self.analysis = [], []
            for wl in worklist:
                self.repair(wl)
            for al in analysis:
                self.repair_anaylsis(al)
        return

    def repair(self, wl):
        '''
        Remove the enode from hashcon, canonicalize it
        and check if the canonical is already in the hashcon,
        if it is, merge, if it is not, add it to hashcon
        '''
        enode, eclass_id = wl
        self.hashcon.pop(enode, None)
        enode = self.canonicalize(enode)
        eclass_id = self.find(eclass_id)
        if enode in self.hashcon:
            self.merge(self.hashcon[enode], eclass_id) # this can insert new items to worklist
        else:
            self.hashcon[enode] = eclass_id
        return

    def repair_anaylsis(self, al):
        '''
        join the data of an e-class with the data obtained by an e-node.
        This is not fully implemented since we need to implement joinA and modifyA
        '''
        enode, eclass_id = al
        canon_id = self.find(eclass_id)
        eclass = self.map_class[canon_id]
        new_data = eclass.eclass_data  # joinA (eclass.eclass_data, makeA(enode))
        if eclass.eclass_data != new_data:
            self.analysis = eclass.parents + self.analysis
            self.map_class[canon_id].eclass_data = new_data
            # modifyA(canon_id)


class EClass:
    '''
    An e-class contains an id and a set of enodes.
    These enodes evaluates to equivalent expressions, so following any
    e-node of the same e-class have the same results.

    Eg.:
    e-class: 1
    e-nodes: {Add(x0, x0), Mul(2.0, x0)}

    The eclass_data field contains a numerical value in case this eclass evaluates to a constant.
 
    The parents contains a list of e-nodes that points to this e-class. This is used during the merge of two e-classes
    '''
    def __init__(self, cid, enode, consts=[], height=0, cache=None):
        self.id = cid
        self.enodes = set([enode])
        match enode:
            case _:
                self.eclass_data = consts
        self.parents = []
        self.height = height
        self.cache = cache

def expr_to_egraph(expr, egraph):
    '''
    Create an e-graph from an expr.

    For each child of the root node, update egraph, replace the child
    with the e-class id and add the node.
    '''
    new_children = []
    for child in children(expr):
        new_children.append(expr_to_egraph(child, egraph))
    n = replaceChildren(expr, new_children)
    return egraph.add(n)
