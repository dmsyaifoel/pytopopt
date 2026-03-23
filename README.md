# pyTOPOPT
# Topology optimization in pure Python for educational purposes

Dependencies: numpy, scipy, matplotlib

USAGE: each .py file is standalone.
* default.py is thoroughly commented, but quite slow.
* optimized.py has been optimized for speed and line count (46) using numpy magic, but is otherwise equivalent.
* adaptivegradientdescent.py uses a novel constraint satisfaction algorithm to add support for many constraints, and yields fast but inaccurate results.
* moto.py uses the pymoto framework for experimental work, which is continuously updated.
