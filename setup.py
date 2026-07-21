from __future__ import annotations

import pybind11
from setuptools import Extension, setup

extensions = [
    Extension(
        "swift_spiral_ics.physics._grid_solver_cpp",
        ["src/swift_spiral_ics/physics/grid_solver_cpp.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3"],
    )
]


setup(ext_modules=extensions)
