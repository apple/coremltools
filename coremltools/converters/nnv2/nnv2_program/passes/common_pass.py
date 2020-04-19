from coremltools.converters.nnv2.nnv2_program.passes.pass_registry import PASS_REGISTRY
import logging


def common_pass(prog):
    passes = [
        'common::const_elimination',
        'common::matmul_to_linear',
        'common::const_elimination',
        'common::loop_invariant_elimination',
        'common::remove_symbolic_reshape',
        'common::reduce_transposes',
        'common::dead_code_elimination', # always end with dce
    ]

    logging.debug('Program before common passes:\n{}'.format(prog))

    prog.validate()
    for p in passes:
        logging.info('Performing pass: "{}"'.format(p))
        PASS_REGISTRY[p](prog)
        prog.validate()

    logging.debug('Program after common passes:\n{}'.format(prog))