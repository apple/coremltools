# -*- coding: utf-8 -*-

#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from functools import partial
import itertools

from ...mil.passes import pass_registry

# IMPORTANT: List of Asssumptions we are making about the problem
# 1) The user defined pattern has exactly one root variable, and one final output operation. As such, we will be searching for a singlular
#    root variable in the larger program, and using that root variable as a starting point for our pattern matching.
#    And, we will only match one of the final operations for the larger program.
# 2) The root variable in the larger program, where we start off the pattern matching, must have the same number of child ops as the
#    root variable in the user defined program
# 3) The outputs of an operation are stored in identical, predictable order. The child operations of an operation are stored in a random order.



class Pattern:

    """This class will have references to all the ops that we have captured in the main, larger program.
        Each captured op will be an attribute of this class. The attribute name will be the same name
        that the user defined in their pattern. So, if the user defines a pattern add(name = 'add_1') -> sub(name = 'sub_1),
        the pattern object will have the fields pattern.add_1, pattern.sub_1, which are references to the corresponding operations
        in the larger program.


        Minimum Attributes:
        root_var: which is the root variable of the first operation of the captured pattern (and corresponds to the user defined pattern’s root variable)
        final_op: the operation in the larger machine learning model that corresponds to the last operation in the user defined pattern.
        block: the block in the larger machine learning model where the pattern was found
        op_set: a set of all the operations captured from the larger machine learning model
        attribute_set: used for enforcing naming (ie, so the user doesn't overwrite any of the variables mentioned above)

        Setters
        set_root_var(root_var): sets the root_var attribute of the Pattern with the given root_var
        set_block(block): sets the block attribute of the Pattern with the given block
        set_final_op(op_name, final_op): adds the operation in question to the pattern and also sets it as the final_op

        Other Methods
        add_attribute(attribute_name, attribute): Adds an attribute to the pattern object. Can be useful for the user.
                                                  Verifies name using the attribute set mentioned above
        add_op(op_name, op): Adds an operation to the pattern, as an attribute which can be accessed and as part of the op_set
        op_list(): convers the op_set to a list and returns it to make it easier for the user

    """

    def __init__(self):
        self.root_var = None
        self.block = None
        self.final_op = None
        self.op_set = set()
        self.attribute_set = set(["root_var", "block", "final_op", "op_set", "attribute_set"])

    def set_root_var(self, root_var):
        self.root_var = root_var

    def set_block(self, block):
        self.block = block

    def set_final_op(self, op_name, final_op):
        self.add_op(op_name, final_op)
        self.final_op = final_op

    def add_attribute(self, attribute_name, attribute):
        if attribute_name in self.attribute_set:
            raise NameError("Pattern " + attribute_name + " is being overwritten. "
                "Make sure every operation in your MIL pattern to detect "
                "has a unique name, and that no operation in it or an attribute you are setting is named "
                "root_var, block, final_op, op_set, or attribute_set.")
        setattr(self, attribute_name, attribute)

    def add_op(self, op_name, op):
        self.add_attribute(op_name, op)
        self.op_set.add(op)

    def op_list(self):
        return list(self.op_set)

def _lists_op_equality(oplist1, oplist2):
    if (len(oplist1) != len(oplist2)):
        return False;

    for i in range(len(oplist1)):
        if oplist1[i].op_type != oplist2[i].op_type:
            return False

    return True

def _pattern_detected(pattern, program_op, pattern_op, program_root_var, pattern_root_var, block):
    # If the pattern_op is None, that means we are dealing with root_var checking (which don't have op_types or outputs)
    if pattern_op is not None and program_op.op_type != pattern_op.op_type:
        return False

    if pattern_op is not None and len(program_op.outputs) != len(pattern_op.outputs):
        return False

    for i in range(len(program_op.outputs) if pattern_op is not None else 1):
        output_same = False

        # ASSUMTION: Assumming that the outputs of an operation are ordered in a particular way
        # So, two identical operations will have the same ordering of outputs.
        program_child_op_list = list(program_op.outputs[i].child_ops) if pattern_op is not None else program_root_var.child_ops
        pattern_child_op_list = list(pattern_op.outputs[i].child_ops) if pattern_op is not None else pattern_root_var.child_ops

        # Last op in the pattern
        if len(pattern_child_op_list) == 0:
            if pattern.final_op is not None and pattern.final_op != program_op:
                raise ValueError("User defined pattern has more than one final operation")
            pattern.set_final_op(pattern_op.name, program_op)
            return True

        if len(program_child_op_list) != len(pattern_child_op_list):
            return False

        # Permuting the program child operations so that at least one of the permutations will be in
        # the exact same order as the pattern child operations
        op_combos = list(itertools.permutations(pattern_child_op_list))

        for combo in op_combos:
            if _lists_op_equality(combo, program_child_op_list):
                truly_equal = True

                for i in range(len(combo)):
                    truly_equal = truly_equal and _pattern_detected(pattern, program_child_op_list[i], combo[i], program_root_var, pattern_root_var, block)

                if truly_equal:
                    # The operations in this sequence match perfectly with the pattern
                    output_same = True
                    break

        if output_same == False:
            return False

    if pattern_op is not None:
        pattern.add_op(pattern_op.name, program_op)
    return True


# This function finds the root_variable in the program that matches with the root_variable in the pattern,
# And then kicks off the pattern matching from there
def _detect_pattern(program_op, ops_arrangement_root_var, block):
    # The goal of this function is to find the root variable of both operations
    program_op_inputs = program_op.get_flattened_inputs()

    for potential_program_root_variable in program_op_inputs:
        pattern = Pattern()
        pattern.set_block(block)

        if _pattern_detected(pattern, program_op, ops_arrangement_root_var.op, potential_program_root_variable, ops_arrangement_root_var, block):
            pattern.set_root_var(potential_program_root_variable)

            # check that none of the ops in this pattern is connected to the output
            # (except the last one)
            for op in pattern.op_list():
               if op is not pattern.final_op:
                    for out in op.outputs:
                        if out in pattern.block.outputs:
                            return False, None

            return True, pattern

    return False, None


def _fuse_one_block(block, ops_arrangement, var_constraints, transform_pattern):
    fusion_status = False
    for op in list(block.operations):
        for b in op.blocks:
            block_changed = True
            while block_changed:
                block_changed = _fuse_one_block(b, ops_arrangement, var_constraints, transform_pattern)

        with block:
            ops_arrangement_root_var = list(ops_arrangement.functions.values())[0].function_inputs[0]
            fusion_status, pattern = _detect_pattern(op, ops_arrangement_root_var, block)

            if fusion_status:
                fusion_status &= var_constraints(pattern)

            if fusion_status:
                transform_pattern(pattern)
                return fusion_status

    return fusion_status


def _fuse_all_blocks(ops_arrangement, var_constraints, transform_pattern, prog):
    for f in prog.functions.values():
        block_changed = True
        while block_changed:
            block_changed = _fuse_one_block(f, ops_arrangement, var_constraints, transform_pattern)


class PassContainer():
    def __init__(self, pass_name):
        self.pass_name = pass_name
        self.passes = []

    def __call__(self, prog):
        if len(self.passes) == 0:
            raise ValueError("no pass functions associated with " + self.pass_name)

        for one_pass in self.passes:
            one_pass(prog)
            prog.validate()

    def add(self, pass_function):
        self.passes.append(pass_function)

def register_generic_pass(ops_arrangement, var_constraints, transform_pattern, pass_name, namespace):
    pass_function = partial(_fuse_all_blocks, ops_arrangement, var_constraints, transform_pattern)

    pass_id = namespace + "::" + pass_name
    if pass_id not in pass_registry.PASS_REGISTRY or not isinstance(pass_registry.PASS_REGISTRY[pass_id], PassContainer):
        pass_registry.PASS_REGISTRY.passes[pass_id] = PassContainer(pass_name)

    pass_registry.PASS_REGISTRY[pass_id].add(pass_function)
