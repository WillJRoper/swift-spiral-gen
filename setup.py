# setup.py for pybind11 extension
import os
import sys
import platform
import subprocess
import re
import site

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r"version\s*([\d.]+)", subprocess.check_output(["cmake", "--version"]).decode()).group(1)
            )
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for all projects
        cfg = "Debug" if self.debug else "Release"
        # CMAKE_ARGS = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}'.format(extdir),
        #               '-DPYTHON_EXECUTABLE={}'.format(sys.executable),
        #               '-DCMAKE_BUILD_TYPE={}'.format(cfg)]
        # This will be passed to `cmake`
        # Find pybind11 install directory
        pybind11_cmake_dir = ""
        try:
            pybind11_cmake_dir = subprocess.check_output(
                [sys.executable, "-m", "pybind11", "--cmakedir"], text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback if pybind11 --cmakedir is not found (older versions or custom install)
            # Try common install locations within site-packages
            for sp in site.getsitepackages():
                test_path = os.path.join(sp, 'pybind11', 'share', 'cmake', 'pybind11')
                if os.path.exists(test_path):
                    pybind11_cmake_dir = test_path
                    break
            if not os.path.exists(pybind11_cmake_dir):
                raise RuntimeError("Could not find pybind11 CMake directory. Please ensure pybind11 is installed.")

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-Dpybind11_DIR={pybind11_cmake_dir}", # Pass pybind11 CMake directory
        ]
        build_args = []

        if platform.system() == "Windows":
            cmake_args += ["-DCMAKE_GENERATOR_PLATFORM={}".format(platform.architecture()[0])]
            if self.debug:
                build_args += ["--config", "Debug"]
            build_args += ["--", "/m"]
        else:
            build_args += ["--", "-j4"]

        env = os.environ.copy()
        env["CXXFLAGS"] = f'{env.get("CXXFLAGS", "")} -DVERSION_INFO=\"{self.distribution.get_version()}\"'
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp, env=env
        )
        print() # Add an empty line for cleaner output


setup(
    name="swift-spiral-ics",
    version="0.1.0",
    author="SWIFT Spiral ICs Team",
    description="Python bindings for C++ grid solver",
    long_description="",
    ext_modules=[CMakeExtension("swift_spiral_ics.physics._grid_solver_cpp")],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
