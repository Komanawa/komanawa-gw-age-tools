"""
created matt_dumont 
on: 16/05/24
"""
from komanawa.ksl_tools.code_optimisation_tools.timeit_analysis import timeit_test
from pathlib import Path

if __name__ == '__main__':
    timeit_test(Path(__file__).parent.joinpath('lightweight_timetest_base.py'),
                function_names=('lightweight_v6', 'lightweight_v5','lightweight_v5','lightweight_v6',), n=100)
