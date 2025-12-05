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
class Mul:
    left: Any
    right: Any

@dataclass(unsafe_hash=True)
class Div:
    left: Any
    right: Any

@dataclass(unsafe_hash=True)
class Const:
    value: float

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
    | Mul
    | Div
    | Exp
    | Log
    | Var
    | Const
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
        # Binary operations
        case Add(l, r):
            return "(" + showTree(l) + ") + (" + showTree(r) + ")"
        case Sub(l, r):
            return "(" + showTree(l) + ") - (" + showTree(r) + ")"
        case Mul(l, r):
            return "(" + showTree(l) + ") * (" + showTree(r) + ")"
        case Div(l, r):
            return "(" + showTree(l) + ") / (" + showTree(r) + ")"
        case AbsoluteDifference2(l, r):
            return f"AbsoluteDifference2({showTree(l)}, {showTree(r)})"
        case BitwiseAnd(l, r):
            return f"BitwiseAnd({showTree(l)}, {showTree(r)})"
        case BitwiseAndMask(l, r):
            return f"BitwiseAndMask({showTree(l)}, {showTree(r)})"
        case BitwiseOr(l, r):
            return f"BitwiseOr({showTree(l)}, {showTree(r)})"
        case BitwiseXor(l, r):
            return f"BitwiseXor({showTree(l)}, {showTree(r)})"
        case Min(l, r):
            return f"Min({showTree(l)}, {showTree(r)})"
        case Max(l, r):
            return f"Max({showTree(l)}, {showTree(r)})"
        case Mean(l, r):
            return f"Mean({showTree(l)}, {showTree(r)})"
        # Unary operations
        case Exp(x):
            return f"exp({showTree(x)})"
        case Log(x):
            return f"log({showTree(x)})"
        case Erode(x):
            return f"Erode({showTree(x)})"
        case Dilate(x):
            return f"Dilate({showTree(x)})"
        case Open(x):
            return f"Open({showTree(x)})"
        case Close(x):
            return f"Close({showTree(x)})"
        case MorphGradient(x):
            return f"MorphGradient({showTree(x)})"
        case MorphTopHat(x):
            return f"MorphTopHat({showTree(x)})"
        case MorphBlackHat(x):
            return f"MorphBlackHat({showTree(x)})"
        case FillHoles(x):
            return f"FillHoles({showTree(x)})"
        case RemoveSmallHoles(x):
            return f"RemoveSmallHoles({showTree(x)})"
        case RemoveSmallObjects(x):
            return f"RemoveSmallObjects({showTree(x)})"
        case MedianBlur(x):
            return f"MedianBlur({showTree(x)})"
        case GaussianBlur(x):
            return f"GaussianBlur({showTree(x)})"
        case Laplacian(x):
            return f"Laplacian({showTree(x)})"
        case Sobel(x):
            return f"Sobel({showTree(x)})"
        case RobertCross(x):
            return f"RobertCross({showTree(x)})"
        case Canny(x):
            return f"Canny({showTree(x)})"
        case Sharpen(x):
            return f"Sharpen({showTree(x)})"
        case Kirsch(x):
            return f"Kirsch({showTree(x)})"
        case Embossing(x):
            return f"Embossing({showTree(x)})"
        case Pyr(x):
            return f"Pyr({showTree(x)})"
        case Denoizing(x):
            return f"Denoizing({showTree(x)})"
        case AbsoluteDifference(x):
            return f"AbsoluteDifference({showTree(x)})"
        case RelativeDifference(x):
            return f"RelativeDifference({showTree(x)})"
        case FluoTopHat(x):
            return f"FluoTopHat({showTree(x)})"
        case GaborFilter(x):
            return f"GaborFilter({showTree(x)})"
        case DistanceTransform(x):
            return f"DistanceTransform({showTree(x)})"
        case DistanceTransformAndThresh(x):
            return f"DistanceTransformAndThresh({showTree(x)})"
        case Threshold(x):
            return f"Threshold({showTree(x)})"
        case ThresholdAt1(x):
            return f"ThresholdAt1({showTree(x)})"
        case BinaryInRange(x):
            return f"BinaryInRange({showTree(x)})"
        case InRange(x):
            return f"InRange({showTree(x)})"
        case BitwiseNot(x):
            return f"BitwiseNot({showTree(x)})"
        case SquareRoot(x):
            return f"SquareRoot({showTree(x)})"
        case Square(x):
            return f"Square({showTree(x)})"
        case ThresholdOtsu(x):
            return f"ThresholdOtsu({showTree(x)})"
        case ContourArea(x):
            return f"ContourArea({showTree(x)})"
        # Leaf nodes
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
        # Binary operations
        case Add(l, r):
            return Add(*cs)
        case Sub(l, r):
            return Sub(*cs)
        case Mul(l, r):
            return Mul(*cs)
        case Div(l, r):
            return Div(*cs)
        case AbsoluteDifference2(l, r):
            return AbsoluteDifference2(*cs)
        case BitwiseAnd(l, r):
            return BitwiseAnd(*cs)
        case BitwiseAndMask(l, r):
            return BitwiseAndMask(*cs)
        case BitwiseOr(l, r):
            return BitwiseOr(*cs)
        case BitwiseXor(l, r):
            return BitwiseXor(*cs)
        case Min(l, r):
            return Min(*cs)
        case Max(l, r):
            return Max(*cs)
        case Mean(l, r):
            return Mean(*cs)
        # Unary operations
        case Exp(n):
            return Exp(*cs)
        case Log(n):
            return Log(*cs)
        case Erode(n):
            return Erode(*cs)
        case Dilate(n):
            return Dilate(*cs)
        case Open(n):
            return Open(*cs)
        case Close(n):
            return Close(*cs)
        case MorphGradient(n):
            return MorphGradient(*cs)
        case MorphTopHat(n):
            return MorphTopHat(*cs)
        case MorphBlackHat(n):
            return MorphBlackHat(*cs)
        case FillHoles(n):
            return FillHoles(*cs)
        case RemoveSmallHoles(n):
            return RemoveSmallHoles(*cs)
        case RemoveSmallObjects(n):
            return RemoveSmallObjects(*cs)
        case MedianBlur(n):
            return MedianBlur(*cs)
        case GaussianBlur(n):
            return GaussianBlur(*cs)
        case Laplacian(n):
            return Laplacian(*cs)
        case Sobel(n):
            return Sobel(*cs)
        case RobertCross(n):
            return RobertCross(*cs)
        case Canny(n):
            return Canny(*cs)
        case Sharpen(n):
            return Sharpen(*cs)
        case Kirsch(n):
            return Kirsch(*cs)
        case Embossing(n):
            return Embossing(*cs)
        case Pyr(n):
            return Pyr(*cs)
        case Denoizing(n):
            return Denoizing(*cs)
        case AbsoluteDifference(n):
            return AbsoluteDifference(*cs)
        case RelativeDifference(n):
            return RelativeDifference(*cs)
        case FluoTopHat(n):
            return FluoTopHat(*cs)
        case GaborFilter(n):
            return GaborFilter(*cs)
        case DistanceTransform(n):
            return DistanceTransform(*cs)
        case DistanceTransformAndThresh(n):
            return DistanceTransformAndThresh(*cs)
        case Threshold(n):
            return Threshold(*cs)
        case ThresholdAt1(n):
            return ThresholdAt1(*cs)
        case BinaryInRange(n):
            return BinaryInRange(*cs)
        case InRange(n):
            return InRange(*cs)
        case BitwiseNot(n):
            return BitwiseNot(*cs)
        case SquareRoot(n):
            return SquareRoot(*cs)
        case Square(n):
            return Square(*cs)
        case ThresholdOtsu(n):
            return ThresholdOtsu(*cs)
        case ContourArea(n):
            return ContourArea(*cs)
        # Leaf nodes - no children to replace
        case _ as n:
            return n

def children(t: Expr) -> [Any]:
    match t:
        # Binary operations
        case Add(l, r) | Sub(l, r) | Mul(l, r) | Div(l, r):
            return [l, r]
        case AbsoluteDifference2(l, r) | BitwiseAnd(l, r) | BitwiseAndMask(l, r):
            return [l, r]
        case BitwiseOr(l, r) | BitwiseXor(l, r) | Min(l, r) | Max(l, r) | Mean(l, r):
            return [l, r]
        # Unary operations
        case Log(n) | Exp(n) | Erode(n) | Dilate(n) | Open(n) | Close(n):
            return [n]
        case MorphGradient(n) | MorphTopHat(n) | MorphBlackHat(n) | FillHoles(n):
            return [n]
        case RemoveSmallHoles(n) | RemoveSmallObjects(n) | MedianBlur(n) | GaussianBlur(n):
            return [n]
        case Laplacian(n) | Sobel(n) | RobertCross(n) | Canny(n) | Sharpen(n):
            return [n]
        case Kirsch(n) | Embossing(n) | Pyr(n) | Denoizing(n) | AbsoluteDifference(n):
            return [n]
        case RelativeDifference(n) | FluoTopHat(n) | GaborFilter(n) | DistanceTransform(n):
            return [n]
        case DistanceTransformAndThresh(n) | Threshold(n) | ThresholdAt1(n) | BinaryInRange(n):
            return [n]
        case InRange(n) | BitwiseNot(n) | SquareRoot(n) | Square(n):
            return [n]
        case ThresholdOtsu(n) | ContourArea(n):
            return [n]
        # Leaf nodes - no children
        case _:
            return []

def operator(t: Expr) -> Expr:
    return applyTree(lambda x: None, t)

def costFun(n):
    match n:
        # Arithmetic operations - lower cost
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
        # Leaf nodes - minimal cost
        case Const(x) | Var(x):
            return 1
        # Binary image processing operations - moderate cost
        case AbsoluteDifference2(l, r) | BitwiseAnd(l, r) | BitwiseAndMask(l, r):
            return 2
        case BitwiseOr(l, r) | BitwiseXor(l, r) | Min(l, r) | Max(l, r) | Mean(l, r):
            return 2
        # Unary morphological operations - moderate cost
        case Erode(x) | Dilate(x) | Open(x) | Close(x):
            return 3
        case MorphGradient(x) | MorphTopHat(x) | MorphBlackHat(x):
            return 3
        case FillHoles(x) | RemoveSmallHoles(x) | RemoveSmallObjects(x):
            return 3
        # Blur and filter operations - moderate cost
        case MedianBlur(x) | GaussianBlur(x) | Laplacian(x):
            return 3
        case Sobel(x) | RobertCross(x) | Sharpen(x):
            return 3
        # More complex edge detectors and filters - higher cost
        case Canny(x) | Kirsch(x) | Embossing(x):
            return 4
        case Pyr(x) | Denoizing(x):
            return 4
        # Specialized operations - higher cost
        case AbsoluteDifference(x) | RelativeDifference(x) | FluoTopHat(x):
            return 3
        case GaborFilter(x) | DistanceTransform(x) | DistanceTransformAndThresh(x):
            return 4
        # Threshold operations - moderate cost
        case Threshold(x) | ThresholdAt1(x) | BinaryInRange(x) | InRange(x):
            return 2
        case ThresholdOtsu(x):
            return 3
        # Bit operations
        case BitwiseNot(x):
            return 1
        # Math operations
        case SquareRoot(x):
            return 2
        case Square(x):
            return 2
        # Complex analysis
        case ContourArea(x):
            return 4
        # Default for unmatched patterns
        case _:
            return 0
