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
class Sub:
    left: Any
    right: Any

@dataclass(unsafe_hash=True)
class Mul:
    left: Any
    right: Any

@dataclass(unsafe_hash=True)
class Div:
    left: Any
    right: Any

@dataclass(unsafe_hash=True)
class Exp:
    child: Any

@dataclass(unsafe_hash=True)
class Log:
    child: Any

@dataclass(unsafe_hash=True)
class Sqrt:
    child: Any

@dataclass(unsafe_hash=True)
class Pow:
    left: Any
    right: Any

@dataclass(unsafe_hash=True)
class Min:
    left: Any
    right: Any

@dataclass(unsafe_hash=True)
class Max:
    left: Any
    right: Any

@dataclass(unsafe_hash=True)
class Mean:
    left: Any
    right: Any

@dataclass(unsafe_hash=True)
class Var:
    idx: int


@dataclass(unsafe_hash=True)
class Const:
    val: float


Expr = Add | Sub | Mul | Div | Log | Exp | Sqrt | Pow | Min | Max | Mean | Var | Const

def showTree(t: Expr) -> str:
    match t:
        case Add(l, r):
            return "(" + showTree(l) + ") + (" + showTree(r) + ")"
        case Sub(l, r):
            return "(" + showTree(l) + ") - (" + showTree(r) + ")"
        case Div(l, r):
            return "(" + showTree(l) + ") / (" + showTree(r) + ")"
        case Mul(l, r):
            return "(" + showTree(l) + ") * (" + showTree(r) + ")"
        case Log(x):
            return f"log(" + showTree(x) + ")"
        case Exp(x):
            return f"exp(" + showTree(x) + ")"
        case Sqrt(x):
            return f"sqrt(" + showTree(x) + ")"
        case Pow(l, r):
            return "(" + showTree(l) + ") ** (" + showTree(r) + ")"
        case Min(l, r):
            return "min(" + showTree(l) + ", " + showTree(r) + ")"
        case Max(l, r):
            return "max(" + showTree(l) + ", " + showTree(r) + ")"
        case Mean(l, r):
            return "mean(" + showTree(l) + ", " + showTree(r) + ")"
        case Var(x):
            return f"X{x}"
        case Const(x):
            return f"{x}"
        case _ as unreachable:
            return str(unreachable) # assert_never(unreachable)

def applyTree(f: Callable, t: Expr) -> Expr:
    new_children = [f(c) for c in children(t)]
    return replaceChildren(t, new_children)

def replaceChildren(t: Expr, cs: [Any]) -> Expr:
    match t:
        case Add(l, r):
            return Add(*cs)
        case Mul(l, r):
            return Mul(*cs)
        case Sub(l, r):
            return Sub(*cs)
        case Div(l, r):
            return Div(*cs)
        case Exp(n):
            return Exp(*cs)
        case Log(n):
            return Log(*cs)
        case Sqrt(n):
            return Sqrt(*cs)
        case Pow(l, r):
            return Pow(*cs)
        case Min(l, r):
            return Min(*cs)
        case Max(l, r):
            return Max(*cs)
        case Mean(l, r):
            return Mean(*cs)
        case _ as n:
            return n

def children(t: Expr) -> [Any]:
    match t:
        case Add(l, r) | Sub(l, r) | Mul(l, r) | Div(l, r) | Pow(l, r) | Min(l, r) | Max(l, r) | Mean(l, r):
            return [l, r]
        case Log(n) | Exp(n) | Sqrt(n):
            return [n]
        case _:
            return []

def operator(t: Expr) -> Expr:
    return applyTree(lambda x: None, t)

def costFun(n):
    match n:
        case Add(l, r):
            return 1
        case Sub(l, r):
            return 1
        case Mul(l, r):
            return 2
        case Div(l, r):
            return 3
        case Log(n):
            return 2
        case Exp(n):
            return 3
        case Sqrt(n):
            return 2
        case Pow(l, r):
            return 3
        case Min(l, r):
            return 2
        case Max(l, r):
            return 2
        case Mean(l, r):
            return 2
        case Const(x) | Var(x):
            return 1
        case _:
            return 0