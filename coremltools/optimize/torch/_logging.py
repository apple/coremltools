#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import logging
import os


def init_root_logger():
    logger = get_root_logger()
    logger.propagate = False
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.addHandler(logging.StreamHandler())
    level = os.environ.get("COREMLTOOLS_OPTIMIZE_TORCH_LOG_LEVEL", "info").upper()
    logger.setLevel(level)
    set_logger_formatter(logger)
    return logger


def get_root_logger():
    return logging.getLogger("coremltools.optimize.torch")


def set_logger_formatter(logger, rank=None):
    rank_component = f"rank {rank}:" if rank is not None else ""
    fmt = f"{rank_component}%(asctime)s:%(name)s:%(lineno)s:%(levelname)s: %(message)s"
    formatter = logging.Formatter(fmt=fmt)
    for handler in logger.handlers:
        handler.setFormatter(formatter)


def set_logger_filters(logger, rank=None):
    for handler in logger.handlers:
        handler.addFilter(RankZeroFilter(rank))


def set_rank_for_root_logger(rank):
    logger = get_root_logger()
    set_logger_formatter(logger, rank)
    set_logger_filters(logger, rank)


class RankZeroFilter(logging.Filter):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        return self.rank == 0
