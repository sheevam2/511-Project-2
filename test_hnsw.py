import pytest
import numpy as np
import os

def test_part0_output():
    """
    Test Part 0: Verify that output.txt exists and contains the correct 10 indices.
    """
    # Check if output.txt exists in part0 directory
    output_path = os.path.join("part0", "output.txt")
    
    assert os.path.exists(output_path), f"output.txt not found at {output_path}. Make sure Part 0 script has been run."
    
    # Read and verify the output
    expected = [932085, 934876, 561813, 708177, 706771, 695756, 435345, 701258, 455537, 872728]
    
    with open(output_path, "r") as output_file:
        output_lines = output_file.readlines()
    
    # Clean up lines and convert to integers
    output_lines = [int(index.strip()) for index in output_lines if index.strip()]
    
    # Verify we have exactly 10 indices
    assert len(output_lines) == 10, f"Expected 10 indices, got {len(output_lines)}"
    
    # Verify the indices match (order matters for Part 0)
    assert output_lines == expected, f"Output does not match expected output. Got {output_lines}, expected {expected}"
    
