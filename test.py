"""
Equality saturation in Python 3.10+
"""

# Author: Fabricio Olivetti de Franca <folivetti@ufabc.edu.br>
#
# License: MIT

from expr import *
from db import *
from egraph import *

tree = Add(Var(0), Mul(Const(1.2), Var(0)))
tree0 = applyTree(lambda x: 0, tree)

pattern = Pattern(Mul("x", Add("y", "z")), Add(Mul("x", "y"), Mul("x", "z")))
egraph = EGraph()
expr_to_egraph(tree, egraph)
db = DataBase(egraph)
query, root = compileToQuery(pattern.source)
genericJoin(db, query)

print(showTree(tree))
print(tree0)
print(children(tree0))
print(query.vars, [a.classIdOrVar for a in query.atoms])