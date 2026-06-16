import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Code.Enterprise.supervisor.real_symbolic_engine import DeterministicSymbolicEngine

@pytest.fixture
def engine():
    # Map mock tensor indices to semantic types
    type_map = {
        0: "L1_ROOT",
        1: "L1_ROOT_PHYSICS",
        64: "L2_AFFIX",
        65: "L2_AFFIX_POLAR",
        128: "OP_COMPOSE_NOUN",
        129: "OP_INVERT"
    }
    return DeterministicSymbolicEngine(type_map)

def test_verify_composition_valid(engine):
    e_vec = torch.zeros(64); e_vec[0] = 1.0  # L1_ROOT
    p_vec = torch.zeros(64); p_vec[0] = 1.0  # L2_AFFIX (index 64 globally)
    o_vec = torch.zeros(64); o_vec[0] = 1.0  # OP_COMPOSE_NOUN (index 128 globally)
    
    assert engine.verify_composition(e_vec, p_vec, o_vec) == True

def test_verify_composition_invalid_type(engine):
    # Try to compose a physics root with a linguistic affix
    e_vec = torch.zeros(64); e_vec[1] = 1.0  # L1_ROOT_PHYSICS
    p_vec = torch.zeros(64); p_vec[0] = 1.0  # L2_AFFIX 
    o_vec = torch.zeros(64); o_vec[0] = 1.0  # OP_COMPOSE_NOUN
    
    assert engine.verify_composition(e_vec, p_vec, o_vec) == False

def test_verify_ast_math_valid(engine):
    # Valid mathematical expression
    assert engine.verify_equation_ast("mass * acceleration") == True

def test_verify_ast_math_hallucination(engine):
    # Hallucinated/Malformed expression (SyntaxError)
    assert engine.verify_equation_ast("mass * * acceleration +") == False

if __name__ == "__main__":
    pytest.main(["-v", __file__])
