from pathlib import Path
import os
import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

root = Path(__file__).parent
eigen = Path(os.environ.get("MIMO_EIGEN_INCLUDE", str(root / "vendor/eigen-3.4.0")))
if not (eigen / "Eigen/Core").exists():
    raise RuntimeError("Eigen headers missing; run scripts/bootstrap_dependencies.py")

setup(
    ext_modules=[Pybind11Extension(
        "mimo_receiver._core", ["cpp/core.cpp"], include_dirs=[str(eigen)],
        cxx_std=17, extra_compile_args=["-O3"],
        define_macros=[("EIGEN_DONT_PARALLELIZE", "1"), ("EIGEN_USE_BLAS", "1")],
        extra_link_args=["-framework", "Accelerate"] if sys.platform == "darwin" else ["-lblas"],
    )],
    cmdclass={"build_ext": build_ext},
)
