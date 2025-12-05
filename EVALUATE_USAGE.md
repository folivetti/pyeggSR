# Evaluate Function Usage Examples

This document demonstrates how to use the `evaluate` function from `evaluate.py` to evaluate expression trees defined in `expr.py`.

## Basic Usage

```python
import numpy as np
from expr import *
from evaluate import evaluate

# Create some test images
image1 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
image2 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

# Example 1: Simple variable access
expr = Var(0)
result = evaluate(expr, data={0: image1})
# result == image1

# Example 2: Binary operations
expr = Add(Var(0), Var(1))
result = evaluate(expr, data={0: image1, 1: image2})
# result = cv2.add(image1, image2)

# Example 3: Unary operations
expr = BitwiseNot(Var(0))
result = evaluate(expr, data={0: image1})
# result = cv2.bitwise_not(image1)

# Example 4: Nested expressions
expr = Add(Sub(Var(0), Var(1)), BitwiseNot(Var(0)))
result = evaluate(expr, data={0: image1, 1: image2})
# result = cv2.add(cv2.subtract(image1, image2), cv2.bitwise_not(image1))
```

## Function Signature

```python
def evaluate(e: Expr, data=None, const_params=None)
```

### Parameters:
- `e` (Expr): The expression tree to evaluate
- `data` (dict, optional): Dictionary mapping variable indices to their data (numpy arrays)
- `const_params` (list, optional): List of constant parameters for operations that need them (e.g., kernel sizes, thresholds)

### Returns:
- `numpy.ndarray`: The evaluated result (typically a uint8 array for image operations)

## Supported Operations

### Binary Operations
- `Add(left, right)` - Element-wise addition
- `Sub(left, right)` - Element-wise subtraction
- `Min(left, right)` - Element-wise minimum
- `Max(left, right)` - Element-wise maximum
- `Mean(left, right)` - Weighted average
- `BitwiseAnd(left, right)` - Bitwise AND
- `BitwiseOr(left, right)` - Bitwise OR
- `BitwiseXor(left, right)` - Bitwise XOR
- `AbsoluteDifference2(left, right)` - Absolute difference

### Unary Operations

#### Arithmetic
- `Square(child)` - Square each pixel value
- `SquareRoot(child)` - Square root of each pixel value
- `Exp(child)` - Exponential function
- `Log(child)` - Logarithm function

#### Bitwise
- `BitwiseNot(child)` - Bitwise NOT

#### Morphological Operations
- `Erode(child)` - Erosion
- `Dilate(child)` - Dilation
- `Open(child)` - Morphological opening
- `Close(child)` - Morphological closing
- `MorphGradient(child)` - Morphological gradient
- `MorphTopHat(child)` - Top-hat transform
- `MorphBlackHat(child)` - Black-hat transform

#### Filtering and Blurring
- `MedianBlur(child)` - Median blur
- `GaussianBlur(child)` - Gaussian blur
- `Laplacian(child)` - Laplacian edge detection
- `Sobel(child)` - Sobel edge detection
- `Canny(child)` - Canny edge detection
- `Sharpen(child)` - Sharpen filter
- `GaborFilter(child)` - Gabor filter

#### Other Operations
- `Threshold(child)` - Thresholding
- `ThresholdOtsu(child)` - Otsu's thresholding
- `DistanceTransform(child)` - Distance transform
- `FillHoles(child)` - Fill holes in binary image
- `RemoveSmallObjects(child)` - Remove small objects
- `RemoveSmallHoles(child)` - Remove small holes

And many more image processing operations...

## Example: Building Complex Expression Trees

```python
import numpy as np
from expr import *
from evaluate import evaluate

# Create test image
image = np.random.randint(0, 256, (200, 200), dtype=np.uint8)

# Build a complex expression tree:
# Apply Gaussian blur, then compute the difference with the original,
# then apply thresholding
expr = ThresholdOtsu(
    Sub(
        Var(0),
        GaussianBlur(Var(0))
    )
)

result = evaluate(expr, data={0: image}, const_params=[128, 5])
```

## Using const_params

Some operations require constant parameters (e.g., kernel sizes, thresholds). These can be passed via the `const_params` argument:

```python
# Morphological operations often use const_params for kernel configuration
expr = Erode(Var(0))
result = evaluate(expr, data={0: image}, const_params=[128, 5])
# const_params[0] influences kernel shape (RECT, ELLIPSE, or CROSS)
# const_params[1] influences kernel size
```

## Displaying Expression Trees

You can visualize expression trees using the `showTree` function from `expr.py`:

```python
from expr import showTree

expr = Add(Sub(Var(0), Var(1)), BitwiseNot(Var(0)))
print(showTree(expr))
# Output: ((X0) - (X1)) + (BitwiseNot(X0))
```
