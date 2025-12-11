"""
Test suite for the evaluate function.
This demonstrates how to use the evaluate function to evaluate expression trees.
"""

import numpy as np
from expr import *
from egraph import *
from evaluate import *
from readimage import _read_inputs, _read_mask
import cv2 

def test_variable_expression():
    """Test that Var expressions return the correct data."""
    print("Test: Variable expressions")
    img = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    
    expr = Var(0)
    result = evaluate(expr, data={0: img})
    
    assert np.array_equal(result, img), "Var(0) should return the input data"
    print("  ✓ Var expression works correctly")


def test_binary_operations():
    """Test binary operations like Add, Sub, etc."""
    print("\nTest: Binary operations")
    img1 = np.random.randint(0, 128, (50, 50), dtype=np.uint8)
    img2 = np.random.randint(0, 128, (50, 50), dtype=np.uint8)
    
    # Test Add
    expr = Add(Var(0), Var(1))
    result = evaluate(expr, data={0: img1, 1: img2})
    assert result.shape == (50, 50), "Result should have correct shape"
    print("  ✓ Add operation works")
    
    # Test Sub
    expr = Sub(Var(0), Var(1))
    result = evaluate(expr, data={0: img1, 1: img2})
    assert result.shape == (50, 50), "Result should have correct shape"
    print("  ✓ Sub operation works")
    
    # Test Min
    expr = Min(Var(0), Var(1))
    result = evaluate(expr, data={0: img1, 1: img2})
    assert result.shape == (50, 50), "Result should have correct shape"
    print("  ✓ Min operation works")
    
    # Test Max
    expr = Max(Var(0), Var(1))
    result = evaluate(expr, data={0: img1, 1: img2})
    assert result.shape == (50, 50), "Result should have correct shape"
    print("  ✓ Max operation works")


def test_unary_operations():
    """Test unary operations like BitwiseNot, Square, etc."""
    print("\nTest: Unary operations")
    img = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    
    # Test BitwiseNot
    expr = BitwiseNot(Var(0))
    result = evaluate(expr, data={0: img})
    assert result.shape == (50, 50), "Result should have correct shape"
    print("  ✓ BitwiseNot operation works")
    
    # Test Square
    expr = Square(Var(0))
    result = evaluate(expr, data={0: img})
    assert result.shape == (50, 50), "Result should have correct shape"
    print("  ✓ Square operation works")
    
    # Test SquareRoot
    expr = SquareRoot(Var(0))
    result = evaluate(expr, data={0: img})
    assert result.shape == (50, 50), "Result should have correct shape"
    print("  ✓ SquareRoot operation works")


def test_nested_expressions():
    """Test nested/complex expressions."""
    print("\nTest: Nested expressions")
    img1 = np.random.randint(0, 128, (50, 50), dtype=np.uint8)
    img2 = np.random.randint(0, 128, (50, 50), dtype=np.uint8)
    
    # Test: Add(Sub(Var(0), Var(1)), BitwiseNot(Var(0)))
    expr = Add(Sub(Var(0), Var(1)), BitwiseNot(Var(0)))
    result = evaluate(expr, data={0: img1, 1: img2})
    assert result.shape == (50, 50), "Result should have correct shape"
    print("  ✓ Nested expression works")
    
    # Test: Max(Min(Var(0), Var(1)), Var(0))
    expr = Max(Min(Var(0), Var(1)), Var(0))
    result = evaluate(expr, data={0: img1, 1: img2})
    assert result.shape == (50, 50), "Result should have correct shape"
    print("  ✓ Complex nested expression works")


def test_morphological_operations():
    """Test morphological operations with const_params."""
    print("\nTest: Morphological operations")
    # Binary image for morphological operations
    img = np.random.randint(0, 2, (50, 50), dtype=np.uint8) * 255
    
    # Test Erode
    expr = Erode(Var(0))
    result = evaluate(expr, data={0: img}, const_params=[128, 5])
    assert result.shape == (50, 50), "Result should have correct shape"
    print("  ✓ Erode operation works")
    
    # Test Dilate
    expr = Dilate(Var(0))
    result = evaluate(expr, data={0: img}, const_params=[85, 7])
    assert result.shape == (50, 50), "Result should have correct shape"
    print("  ✓ Dilate operation works")


def test_expression_tree_display():
    """Test that expression trees can be displayed."""
    print("\nTest: Expression tree display")
    expr = Add(Sub(Var(0), Var(1)), BitwiseNot(Var(0)))
    tree_str = showTree(expr)
    print(f"  Expression tree: {tree_str}")
    assert isinstance(tree_str, str), "showTree should return a string"
    print("  ✓ Expression tree display works")

def test_read_file():
    img = _read_inputs("NVA_19-002.MelanLizaV10_2.S1528447.P4595.png")
    egraph = EGraph()
    id0 = egraph.add(Var(0))
    id1 = egraph.add(Var(1))
    id2 = egraph.add(Var(2))
    id3 = egraph.add(GaussianBlur(id0), [200])
    id4 = egraph.add(BitwiseOr(id3, id2))
    id5 = egraph.add(BitwiseOr(id4, id1))
    result = evaluate_egraph(id5, egraph, img) 
    cv2.imwrite("test_output.png", result)

if __name__ == "__main__":
    print("=" * 60)
    print("Running evaluate function test suite")
    print("=" * 60)
    
    try:
        test_variable_expression()
        test_binary_operations()
        test_unary_operations()
        test_nested_expressions()
        test_morphological_operations()
        test_expression_tree_display()
        test_read_file()
        print(expr_to_egraph(Add(Var(0), Var(1)), EGraph()))
        
        print("\n" + "=" * 60)
        print("✓ All tests passed successfully!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
