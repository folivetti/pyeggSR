"""
Equality saturation in Python 3.10+
"""

# Author: Fabricio Olivetti de Franca <folivetti@ufabc.edu.br>
#
# License: MIT

from dataclasses import dataclass
from typing import Any, Callable

@dataclass(unsafe_hash=True)
class Add:
    left: Any
    right: Any


@dataclass(unsafe_hash=True)
class Mul:
    left: Any
    right: Any


@dataclass(unsafe_hash=True)
class Var:
    idx: int


@dataclass(unsafe_hash=True)
class Const:
    val: float


Expr = Add | Mul | Var | Const


def showTree(t: Expr) -> str:
    match t:
        case Add(l, r):
            return showTree(l) + " + " + showTree(r)
        case Mul(l, r):
            return showTree(l) + " * " + showTree(r)
        case Var(x):
            return f"X{x}"
        case Const(x):
            return f"{x}"
        case _ as unreachable:
            assert_never(unreachable)

def applyTree(f: Callable, t: Expr) -> Expr:
    match t:
        case Add(l, r):
            return Add(f(l), f(r))
        case Mul(l, r):
            return Mul(f(l), f(r))
        case _ as n:
            return n

def children(t: Expr) -> [Any]:
    match t:
        case Add(l, r) | Mul(l, r):
            return [l, r]
        case _:
            return []

def operator(t: Expr) -> Expr:
    return applyTree(lambda x: None, t)
