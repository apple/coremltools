from coremltools.converters.nnv2.nnv2_program.passes.pass_registry import PASS_REGISTRY
import logging

def tensorflow_passes(prog):
    passes = [
            "common::loop_invariant_elimination",
            "tensorflow::backfill_make_list_elem_type",
    ]

    prog.validate()
    for p in passes:
        logging.info('Performing passes for tf1 frontend: "{}"'.format(p))
        PASS_REGISTRY[p](prog)
        prog.validate()

    logging.debug('Program after tf1 frontend passes:\n{}'.format(prog))
