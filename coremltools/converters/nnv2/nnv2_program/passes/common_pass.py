from coremltools.converters.nnv2.nnv2_program.passes.pass_registry import PASS_REGISTRY
import logging


def common_pass(prog):
    passes = [
        'const_elimination',
        'matmul_to_linear',
        'const_elimination',
        'remove_symbolic_reshape',
        'dead_code_elimination',
    ]

    for p in passes:
        logging.info('Performing common graph pass: "{}"'.format(p))
        PASS_REGISTRY[p](prog)

    logging.debug('SSA after common graph passes:\n{}'.format(prog))
