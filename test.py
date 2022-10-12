"""
Equality saturation in Python 3.10+
"""

# Author: Fabricio Olivetti de Franca <folivetti@ufabc.edu.br>
#
# License: MIT

from expr import *
from db import *
from egraph import *

tree = Mul(Var(0), Add(Mul(Const(1.2), Var(0)), Var(1))) # Add(Var(0), Mul(Const(1.2), Var(0)))
tree0 = applyTree(lambda x: 0, tree)

pattern = Pattern(Mul("x", Add("y", "z")), Add(Mul("x", "y"), Mul("x", "z")))
egraph = EGraph()
expr_to_egraph(tree, egraph)
db = DataBase(egraph)
#print(egraph.hashcon)
#print(db.database)

query, root = compileToQuery(pattern.source)
#print(query.vars, root)
#printAtoms(query.atoms)
join = genericJoin(db, query)

#print(showTree(tree))
#print(tree0)
#print(children(tree0))
#print(query.vars, [a.classIdOrVar for a in query.atoms])
print('==========')
for j in join:
    print(j)