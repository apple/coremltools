from coremltools.converters.nnv2.nnv2_program.passes.pass_registry import PASS_REGISTRY
import logging

def nnv1_backend_passes(prog):
    passes = [
        'nnv1_backend::commingle_loop_vars', # after loop_invariant_elimination
        'nnv1_backend::handle_return_unused_inputs',
        'common::const_elimination',
        'common::dead_code_elimination', # always end with dce
    ]

    for p in passes:
        logging.info('Performing passes for nnv1_backend: "{}"'.format(p))
        PASS_REGISTRY[p](prog)

    logging.debug('Program after nnv1 backend passes:\n{}'.format(prog))
