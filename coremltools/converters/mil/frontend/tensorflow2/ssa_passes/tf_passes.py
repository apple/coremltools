from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
import logging


def tensorflow_passes(prog):
    passes = [
        "common::dead_code_elimination",
        "common::loop_invariant_elimination",

        # tensorflow2::remove_vacuous_cond should come before
        # tensorflow::backfill_make_list_elem_type.
        "tensorflow2::remove_vacuous_cond",
        "tensorflow::backfill_make_list_elem_type",

        # DCE to reduce tf_lstm_block outputs and allow lstm_rewrite to
        # ssa lstm
        "common::dead_code_elimination",
    ]

    prog.validate()
    for p in passes:
        logging.info('Performing passes for TensorFlow 2.x frontend: "{}"'.format(p))
        PASS_REGISTRY[p](prog)
        prog.validate()

    logging.debug('Program after TensorFlow 2.x frontend passes:\n{}'.format(prog))
