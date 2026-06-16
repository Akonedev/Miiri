import ast
import torch

class DeterministicSymbolicEngine:
    """
    Real Symbolic Verification Engine for Miiri.
    Replaces string-matching mocks with deterministic logic parsing.
    """
    def __init__(self, semantic_space_map):
        """
        semantic_space_map maps tensor indices back to semantic types
        for verification.
        Example: {0: "L1_ROOT", 64: "L2_AFFIX", 128: "OP_COMPOSE"}
        """
        self.type_map = semantic_space_map
        
    def _extract_active_types(self, entity_vec, prop_vec, op_vec):
        """Finds the active semantic type by locating the non-zero segment."""
        # Simple extraction for prototype: find argmax and map to type
        e_idx = torch.argmax(entity_vec).item()
        p_idx = torch.argmax(prop_vec).item() + 64 # property offset
        o_idx = torch.argmax(op_vec).item() + 128  # operator offset
        
        e_type = self.type_map.get(e_idx, "UNKNOWN_ENT")
        p_type = self.type_map.get(p_idx, "UNKNOWN_PROP")
        o_type = self.type_map.get(o_idx, "UNKNOWN_OP")
        
        return e_type, p_type, o_type

    def verify_composition(self, entity_vec, prop_vec, op_vec):
        """
        Uses structural typing to verify if a composition is mathematically/linguistically sound.
        """
        e_type, p_type, o_type = self._extract_active_types(entity_vec, prop_vec, op_vec)
        
        # Rule 1: A ROOT can only be composed with an AFFIX using a COMPOSE operator
        if o_type == "OP_COMPOSE_NOUN":
            if e_type == "L1_ROOT" and p_type == "L2_AFFIX":
                return True
            else:
                return False
                
        # Rule 2: Inversion Operator requires a specific property type
        if o_type == "OP_INVERT":
            if p_type == "L2_AFFIX_POLAR":
                return True
            return False

        # If operator is unknown or rules don't match, strictly reject
        return False

    def verify_equation_ast(self, equation_str):
        """
        Demonstrates how Miiri uses Python's AST to verify math formulas
        generated from the latent space before rendering them as tokens.
        """
        try:
            # Parse the string into an Abstract Syntax Tree in eval mode
            # If it parses successfully without a SyntaxError, it is structurally a valid expression.
            ast.parse(equation_str, mode='eval')
            return True
        except SyntaxError:
            # Hallucination detected (malformed math)
            return False
