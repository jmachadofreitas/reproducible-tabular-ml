"""Preprocessing policies for multiple-instance datasets.

MIL preprocessing must respect the bag-level split boundary. Transforms may fit
on instances from training bags, bag-level features, or derived bag summaries,
but never on instances from held-out bags.
"""

from rtml.multi_instance.preprocessing.policies import build_preprocessor

__all__ = ["build_preprocessor"]
