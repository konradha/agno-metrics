import sys, types, os

collect_ignore = [os.path.join(os.path.dirname(__file__), "..", "__init__.py")]

m = types.ModuleType("jax_pde_metric")
m.__path__ = [os.path.dirname(os.path.dirname(__file__))]
sys.modules["jax_pde_metric"] = m
