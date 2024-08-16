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
    '''
    deterministic string hash to replace variables with int
    python's hash is random!
    '''
    acc = 5381
    for c in x:
        acc = 33*acc ^ ord(c)
    return acc

## RULES

class Pattern:
    def __init__(self, source, target):
        '''
        This is actually a Substitution Rule composed of two patterns
        - source: is the pattern that must be matched
        - target: is what the source will be replaced with
        The variables that match anything is a string.

        Example:
            Pattern(Add("x", "y"), Add("y", "x"))
            replace any x+y to y+x, with x and y being any subtree

        TODO: implement the constraints
        '''
        self.source = source
        self.target = target
        self.properties = {}  # maps a var to a tuple of properties (is_not_value, is_value, is_negative, is_not_negative, is_zero,...)

    # TODO: incorporate match into pattern
    def match(self, egraph):
        for enode, eclass_id in egraph.hashcon.items():
            pass

## DATABASE

class DataBase:
    def __init__(self, egraph):
        '''
        The database stores the relation of any operator
        to the classes that contains such operator.

        for example, database[Add] will return a trie
        with all the classes that contains Add.
        To store the operators we replace the childs with None so 
        any two e-nodes with the same operator at the root becomes the same:

            Add(1, 2) and Add(2, 5) becomes Add(None, None) and Add(None, None)
            that are mapped into the same trie.
        
        '''
        self.database = {}  # operator -> IntTrie
        for enode, eclass_id in egraph.hashcon.items():
            self.addToDB(enode, eclass_id)

    def populate(self, trie, ids):
        '''
        

        '''
        if len(ids) == 0:
            ''' if there is nothing to add, just returns the trie '''
            return trie  #IntTrie(None) if trie is None else trie
        if trie is None:
            ''' 
            if the trie still does not exist, creates one with
            the first id (base id) and then adds the child ids 
            as a linked list of tries
            '''
            trie = IntTrie(ids[0])
            tmp = trie
            prev_id = ids[0]
            for next_id in ids[1:]:
                tmp.trie[prev_id] = IntTrie(next_id)
                tmp = tmp.trie[prev_id]
                prev_id = next_id
            #trie.trie[ids[0]] = self.populate(None, ids[1:])
            return trie
        else:
            '''
            If the trie already exists, insert the key into the
            trie if it does not exist, then repeats the process
            for every child.
            '''
            trie.keys |= {ids[0]}
            next_trie = trie.trie.get(ids[1], None)
            trie.trie[ids[0]] = self.populate(next_trie, ids[1:])
            return trie

    def addToDB(self, enode, eclass_id):
        '''
        Adds an e-node operator to the database.

        First we replace the children of the e-node with None,
        retrieve the trie corresponding to that operator (or None if it does not exist)
        and then adds to the database.
        '''
        ids = [eclass_id] + children(enode)
        enode = operator(enode)
        trie = self.database.get(enode, None)
        self.database[enode] = self.populate(trie, ids)
            

class IntTrie:
    def __init__(self, x):
        '''
        The IntTrie stores the e-class ids containing
        the operator as the keys and points to another
        trie that stores the e-class ids of the first child.

        The trie of the first child points to the trie of the
        second child and so on.
        '''
        self.keys = {x}
        self.trie = {}  # {int -> IntTrie()}

## SATURATION
def eqSat(scheduler, expr, rules, costFun):
    '''
    Equality saturation takes an scheduler, an expression,
    a set of rules, and a cost function.

    First we create an EGraph and insert the expression
    into the graph.

    We then run equality saturation for a couple of iterations
    by applying the matched rules at every iteration.

    Finally we extract the best expression using the cost function.
    '''
    egraph = EGraph()
    eclassId = expr_to_egraph(expr, egraph)
    runEqSat(egraph, scheduler, rules)
    return getBest(costFun, egraph, eclassId)[1]

def getBest(costFun, egraph, eclassId):
    def nodeTotCost(node, cd):
        '''
        given a node and a dictionary mapping
        e-class id to a tuple of:
            - the best known cost for an e-class id
            - the best sub-expression for that e-class id
        If the cost of any children is not in
        the dict, we return None as we still
        have to fill it up
        '''
        c = costFun(node)  # calculates the cost of that node
        new_children = []
        for child in children(node):
            # for every child of that node
            if child in cd:
                # retrieve the cost and best sub-tree
                (cc, nc) = cd[child]
                c += cc  # add it to the cost
                new_children.append(nc)  # replace the child with the best sub-expr
            else:
                return None
        return (c, replaceChildren(node, new_children))

    def fillUpCosts(cd, classes):
        '''
        Fills up the dictionary of best cost for a node.
        The strategy here is to find a terminal node and
        calculates the cost upwards.

        classes is a dict that maps a class id to the class object
        '''
        changed = True
        while changed:
            changed = False
            for cid, v in classes.items():
                '''
                For each entry in the dict
                '''
                nodes = v.enodes  # set of nodes of this e-class
                currentCost = cd.get(cid, None)  # current assigned cost for this e-class
                newCost = None
                for n in nodes:
                    '''
                    if newCost is still none, replace with the
                    first non-None cost found.
                    After that, it replaces with a new cost if the
                    new cost is smaller than the current one.
                    '''
                    c = nodeTotCost(n, cd)
                    if newCost is None:
                        newCost = c
                    else:
                        if c is not None:
                            if c[0] < newCost[0]:
                                 newCost = c
                if newCost is not None:
                    '''
                    if we found a non-None new cost:
                        - replaces the current one if it is none
                        - replaces the current one if it is better
                    mark as changed so we go through the entries all over again
                    '''
                    if currentCost is None:
                        cd[cid] = newCost
                        changed = True
                    else:
                        if newCost[0] < currentCost[0]:
                            cd[cid] = newCost
                            changed = True

    '''
    eclassId is the "root" id returned when creating
    the e-graph. It represents all nodes from the root of
    the expression.

    First we call fillUpCosts to store all best costs-expressions
    for each e-class.
    After that, 
    '''
    start_eclass = egraph.find(eclassId)
    costDict = {}
    fillUpCosts(costDict, egraph.map_class)

    '''
    minCD = None
    changed = True
    while changed:
        changed = False
        if start_eclass in costDict:
            if minCD is None:
                minCD = costDict[start_eclass]
                changed = True
            else:
                newCost = costDict[start_eclass]
                if minCD[0] < newCost[0]:
                    minCD = newCost
                    changed = True
        else:
            changed = True
        start_eclass = egraph.union_find[start_eclass]  # not really sure this does anything since start_eclass is canonical
        '''
    # just return the best cost-expr from the start e-class id
    return costDict[start_eclass]  # minCD

def runEqSat(egraph, scheduler, rules):
    rule_sched = {}
    for i in range(30):
        old_map_class = egraph.map_class
        old_hashcon = egraph.hashcon
        db = DataBase(egraph)

        # step 1: match the rules
        matches = []
        for j, r in enumerate(rules):
            m = matchWithScheduler(db, i, rule_sched, j, r)
            matches += m

        # step 2: apply matches
        for match in matches:
            applyMatch(egraph, *match)
        egraph.rebuild()

        new_map_class = egraph.map_class
        new_hashcon = egraph.hashcon
        # if nothing changed, just return
        if (new_map_class == old_map_class) and (new_hashcon == old_hashcon):
            return egraph
    return egraph


def matchWithScheduler(db, i, rule_sched, j, rule):
    # if the rule is banned inside the scheduler, do not return any matches
    if j in rule_sched and rule_sched[j] <= i:
        return []
    # find the matches and ban the rule for 5 iterations
    # this should be customized by the scheduler parameter
    matches = ematch(db, rule.source)
    rule_sched[j] = i + 5
    return [(rule, match) for match in matches]


def isValidConditions(rule, match, egraph):
    return True  # TODO: check the preconditions of the rule


def applyMatch(egraph, rule, match):
    # apply the rule to the matches and merge the
    # newly created e-class
    #
    # match[0] contains the substitution map from
    # VarId to ClassId 
    # match[1] contains the matched class id
    if isValidConditions(rule, match, egraph):
        new_eclass = reprPat(egraph, match[0], rule.target)
        egraph.merge(match[1].cid, new_eclass)


def reprPat(egraph, subst_map, target):
    '''
    traverse the matched subtree replacing the variables
    (represented as strings) with the replacement class ids

    If the current node is a variable, replace it with the
    class id of the substitution map.
    Otherwise, it traverse the children and replace the current
    children with the returned class ids and then add it to the
    e-graph. Adding this new node to the e-graph will create a
    new e-class id that should be merged with the e-class of the pattern.
    '''
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
    def __init__(self, atoms):
        '''
        A Query is a list of atoms, if atoms is empty
        it must match everything
        '''
        self.atoms = atoms


@dataclass(unsafe_hash=True)
class ClassId:
    cid: int


@dataclass(unsafe_hash=True)
class VarId:
    vid: int


# Sum type that can either contain
# a class id or a var id.
ClassOrVar = ClassId | VarId


class Atom:
    def __init__(self, classIdOrVar: ClassOrVar, t: Expr):
        '''
        An atom contains a class-id or var-id identifying
        that this node is related to either one existing
        class id or it is a pattern variable that can be
        replaced by class id.
        It also contains a tree that describes the node
        that was matched with the children pointing to other
        ids
        '''
        self.classIdOrVar = classIdOrVar
        self.relation = t  # tree of classIdOrVar


def ematch(db, source):
    '''
    1. Create a query from the source pattern, returning also the root variable
    2. Find the possible substitutions for the variables
    3. For each substitution map, append the map and the root e-class into the matches
    '''
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
    '''
    If the pattern is a string,
    then it is a pattern var and the query
    is empty (as in, replace with anything)

    Otherwise, sets a state variable 'v' that contains
    the next available var id, and run the auxiliary function
    aux.

    Example:
        The pattern Add("x", "y") would return
         (0, [Atom(0, Add(hash("x"), hash("y"))])
        The pattern Mul("x", Add("y", "z")) would return
         (0, [Atom(0, Mul(hash("x"), 1)), Atom(1, Add(hash("y"), hash("z")))])
    '''
    if isinstance(pat, str):
        return Query([]), pat

    v = 0

    def aux(pat):
        nonlocal v
        if isinstance(pat, str):
            # if it is a pattern variable, returns
            # the variable id using hash and an empty
            # atom list
            return (VarId(hashstr(pat)), [])
        # creates a new var id with the state var 'v'
        rt = VarId(v)
        v = v + 1
        atoms = []
        new_children = []
        # For each child, runs aux on the child
        # replace the children with the returned variables
        # and append the list of atoms.
        for child in children(pat):
            rc, ac = aux(child)
            new_children.append(rc)
            atoms += ac
        # create a new atom with the var id and the node with pat ids
        atoms = [Atom(rt, replaceChildren(pat, new_children))] + atoms
        return (rt, atoms)

    (root, atoms) = aux(pat)
    return Query(atoms), root


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
    # creates all possible substitution maps
    # for a given query
    def genericrec(atoms, variables):
        substs = []
        if len(variables) == 0:
            # if there is no variable to replace
            # create an empty subst map for each atom
            return [{} for _ in atoms]
        # unstack the first variable from the list
        x = variables.pop(0)
        # domainX will return all possible class ids
        # for variable x with the current atoms.
        for classId in domainX(db, x, atoms):
            # replace all occurrence of x with classId
            newatoms = updateAll(x, classId, atoms)
            # generates a list of substitutions for the remaining
            # variables
            for y in genericrec(newatoms, variables.copy()):
                # create the entry x -> classId and append it
                # to the substs list
                y[x] = classId
                substs.append(y)
        return substs

    # sort the variables and apply the rec function
    return genericrec(q.atoms, orderedVars(q))


def domainX(db, var, atoms):

    def elemOfAtom(atom):
        # filter the atoms that contain the
        # variable 'var' as the root or
        # one of the children
        match atom.classIdOrVar:
            case VarId(x):
                return atom.classIdOrVar == var
            case _:
                return any(x == var for x in children(atom.relation))
        return False

    filtered = list(filter(elemOfAtom, atoms))
    return [ClassId(x) for x in intersectAtoms(var, db, filtered)]


def intersectAtoms(var, db, atoms):
    # if there is no atoms, return an empty list
    if len(atoms) == 0:
        return set([])

    def f(atom):
        # if the operator exists in the database
        # get the class ids with intersectTrie using the
        # class-id or var-id of the atom and their children
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

    # apply f to each atom and
    # insert the new class ids into the set
    # with intersection. i.e., the subst must be
    # valid for every atom
    classIds = f(atoms[0])
    for atom in atoms[1:]:
        classIds = classIds & f(atom)
    return classIds


def intersectTrie(var, xs, trie, ids):
    '''
    var: variable current being matched
    xs: dictionary of varid to class id
    trie: current entrie of database 
    ids: list of ids to be intersected 
    '''
    def isDiffFrom(x, y):
        match y:
            case ClassId(z):
                return True
            case VarId(z):
                return VarId(x) != y

    if len(ids) == 0:
        return set([])

    i = ids.pop(0)

    match i:
        case ClassId(x):
            # if it is a class id, look if it exists in
            # the current trie so we can continue the search
            if x in trie.trie:
                return intersectTrie(var, xs, trie.trie[x], ids)
            else:
                return None
        case VarId(x):
            if x in xs:
                # if there is an entry for x, it means that we
                # are currently investigating one class id for x
                # if this class id is in the trie, then it is still valid
                # and we can keep looking at the other ids
                # otherwise, it means it is invalid and we return None (fail)
                if xs[x] in trie.trie:
                    return intersectTrie(var, xs, trie.trie[xs[x]], ids)
                else:
                    return None
            else:
                if VarId(x) == var:
                    # if the current id is the same var id we are
                    # replacing, and there is no other occurrence in this
                    # pattern, we return the class ids associated
                    # with this operator, as they are all valid.
                    if all(map(lambda y: isDiffFrom(x, y), ids)):
                        return trie.keys
                    else:
                        # otherwise for each class-id, trie in the current trie:
                        # - store the class id in xs[x]
                        # - call intersectTrie and if it is not None,
                        # adds the class k into the current list and returns.
                        # Since we are interested in var, we just have to check
                        # whether there is valid substitutions for the remaining variables
                        domains = set([])
                        for k, v in trie.trie.items():
                            xs[x] = k
                            if intersectTrie(var, xs, v, ids) is not None:
                                domains.add(k)
                        return domains
                else:
                    # if it is a different id from what we want to
                    # replace. For each class-id, trie in the current trie:
                    # - store the class id in xs[x]
                    # - call intersectTrie with this association and the next ids
                    # - if it returns something, insert the new class ids into the
                    # current list and return
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
        new_children = [x if c == var else c for c in children(atom.relation)]
        atom.relation = replaceChildren(atom.relation, new_children)
        return atom
    return [replace(atom) for atom in atoms]


def orderedVars(q):
    def elemOfAtom(v, atom):
        match atom.classIdOrVar:
            case VarId(x):
                return atom.classIdOrVar == v
            case _:
                return any(x == v for x in children(atom.relation))
        return False

    def varcost(var):
        '''
        the cost of a variable is: -100*n + sum(len)
        where n is the number of atoms it appears on 
        and sum(len) is the sum of the lengths of the atoms it
        appears on
        '''
        acc = 0
        for atom in q.atoms:
            if elemOfAtom(var, atom):
                acc = acc - 100 + atomLen(atom)
        return acc

    def atomLen(atom):
        return 1 + len(children(atom.relation))

    # generate the list of variables for the atoms
    variables = []
    for atom in q.atoms:
        for a in [atom.classIdOrVar] + children(atom.relation):
            if a not in variables:
                match a:
                    case VarId(x):
                        variables.append(a)
                    case _:
                        pass

    # sort them by the varcost
    return sorted(variables, key=varcost)
