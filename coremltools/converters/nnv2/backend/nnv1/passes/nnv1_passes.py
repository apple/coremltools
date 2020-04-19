from coremltools.converters.nnv2.nnv2_program.passes.pass_registry import PASS_REGISTRY
import logging

def nnv1_backend_passes(prog):
    passes = [
        'nnv1_backend::commingle_loop_vars', # after loop_invariant_elimination
        'nnv1_backend::handle_return_inputs_as_outputs',
        'common::const_elimination',
        'common::dead_code_elimination',
        'nnv1_backend::handle_unused_inputs', # must come after dce.
        'nnv1_backend::alert_return_type_cast', # must be at the end.
    ]

    prog.validate()
    for p in passes:
        logging.info('Performing passes for nnv1_backend: "{}"'.format(p))
        PASS_REGISTRY[p](prog)
        # No more validation from this point on as prog is not SSA anymore.

    logging.debug('Program after nnv1 backend passes:\n{}'.format(prog))
