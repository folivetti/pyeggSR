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
class Exp:
    child: Any

@dataclass(unsafe_hash=True)
class Log:
    child: Any

@dataclass(unsafe_hash=True)
class Var:
    idx: int



# Expanded dataclasses based on pycgp/ipfunctions.py (image-processing ops)
# Many of these are unary (single child) or binary (left/right). Names chosen
# to correspond to the f_* functions in ipfunctions.py but as CamelCase types.

# Morphology / structuring ops (unary)
@dataclass(unsafe_hash=True)
class Erode:
    child: Any

@dataclass(unsafe_hash=True)
class Dilate:
    child: Any

@dataclass(unsafe_hash=True)
class Open:
    child: Any

@dataclass(unsafe_hash=True)
class Close:
    child: Any

@dataclass(unsafe_hash=True)
class MorphGradient:
    child: Any

@dataclass(unsafe_hash=True)
class MorphTopHat:
    child: Any

@dataclass(unsafe_hash=True)
class MorphBlackHat:
    child: Any

@dataclass(unsafe_hash=True)
class FillHoles:
    child: Any

@dataclass(unsafe_hash=True)
class RemoveSmallHoles:
    child: Any

@dataclass(unsafe_hash=True)
class RemoveSmallObjects:
    child: Any

# Blur / filters / edge detectors (unary)
@dataclass(unsafe_hash=True)
class MedianBlur:
    child: Any

@dataclass(unsafe_hash=True)
class GaussianBlur:
    child: Any

@dataclass(unsafe_hash=True)
class Laplacian:
    child: Any

@dataclass(unsafe_hash=True)
class Sobel:
    child: Any

@dataclass(unsafe_hash=True)
class RobertCross:
    child: Any

@dataclass(unsafe_hash=True)
class Canny:
    child: Any

@dataclass(unsafe_hash=True)
class Sharpen:
    child: Any

@dataclass(unsafe_hash=True)
class Kirsch:
    child: Any

@dataclass(unsafe_hash=True)
class Embossing:
    child: Any

@dataclass(unsafe_hash=True)
class Pyr:
    child: Any

@dataclass(unsafe_hash=True)
class Denoizing:
    child: Any

# Arithmetic / comparison like ops (binary)
@dataclass(unsafe_hash=True)
class AbsoluteDifference:
    child: Any

@dataclass(unsafe_hash=True)
class AbsoluteDifference2:
    left: Any
    right: Any

@dataclass(unsafe_hash=True)
class RelativeDifference:
    child: Any

@dataclass(unsafe_hash=True)
class FluoTopHat:
    child: Any

@dataclass(unsafe_hash=True)
class GaborFilter:
    child: Any

@dataclass(unsafe_hash=True)
class DistanceTransform:
    child: Any

@dataclass(unsafe_hash=True)
class DistanceTransformAndThresh:
    child: Any

@dataclass(unsafe_hash=True)
class Threshold:
    child: Any

@dataclass(unsafe_hash=True)
class ThresholdAt1:
    child: Any

@dataclass(unsafe_hash=True)
class BinaryInRange:
    child: Any

@dataclass(unsafe_hash=True)
class InRange:
    child: Any

@dataclass(unsafe_hash=True)
class BitwiseAnd:
    left: Any
    right: Any

@dataclass(unsafe_hash=True)
class BitwiseAndMask:
    left: Any
    right: Any

@dataclass(unsafe_hash=True)
class BitwiseNot:
    child: Any

@dataclass(unsafe_hash=True)
class BitwiseOr:
    left: Any
    right: Any

@dataclass(unsafe_hash=True)
class BitwiseXor:
    left: Any
    right: Any

@dataclass(unsafe_hash=True)
class SquareRoot:
    child: Any

@dataclass(unsafe_hash=True)
class Square:
    child: Any

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
class ThresholdOtsu:
    child: Any

@dataclass(unsafe_hash=True)
class ContourArea:
    child: Any

# Update Expr union to include the expanded operators
Expr = (
    Add
    | Sub
    | Exp
    | Log
    | Var
    | Erode
    | Dilate
    | Open
    | Close
    | MorphGradient
    | MorphTopHat
    | MorphBlackHat
    | FillHoles
    | RemoveSmallHoles
    | RemoveSmallObjects
    | MedianBlur
    | GaussianBlur
    | Laplacian
    | Sobel
    | RobertCross
    | Canny
    | Sharpen
    | Kirsch
    | Embossing
    | Pyr
    | Denoizing
    | AbsoluteDifference
    | AbsoluteDifference2
    | RelativeDifference
    | FluoTopHat
    | GaborFilter
    | DistanceTransform
    | DistanceTransformAndThresh
    | Threshold
    | ThresholdAt1
    | BinaryInRange
    | InRange
    | BitwiseAnd
    | BitwiseAndMask
    | BitwiseNot
    | BitwiseOr
    | BitwiseXor
    | SquareRoot
    | Square
    | Min
    | Max
    | Mean
    | ThresholdOtsu
    | ContourArea
)

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
        case _ as n:
            return n

def children(t: Expr) -> [Any]:
    match t:
        case Add(l, r) | Sub(l,r) | Mul(l, r) | Div(l, r):
            return [l, r]
        case Log(n) | Exp(n):
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
        case Const(x) | Var(x):
            return 1
        case _:
            return 0
