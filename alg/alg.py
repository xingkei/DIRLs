# coding=utf-8
from alg.algs.GMRL import GMRL
from alg.algs.ERM import ERM
from alg.algs.GroupDRO import GroupDRO
from alg.algs.DIFEX import DIFEX

ALGORITHMS = [
    'ERM',
    'GroupDRO',
    'GMRL',
    'DIFEX'
]


def get_algorithm_class(algorithm_name):
    if algorithm_name not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
