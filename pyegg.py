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