#!/usr/bin/env python

import sys
import platform
import subprocess
import os
import numpy as np

# simply assume python3 exists
from setuptools import setup
from setuptools import Extension as _Extension
from Cython.Build import cythonize

# Use global macro instead
# NOTE: we can't use `sys.prefix` because `build` will use isolated,
# temp-created venv that changed the value to `/private/var/xxx`
include_dirs = []
library_dirs = []
extra_link_args = ["-std=c++11"]
cxxflags = list(filter(None, os.environ.get("CXXFLAGS", "").split()))

# using `pkg-config` to test libraries availabiity, before which need to
# set `PKG_CONFIG_PATH` to include where `fplll.pc` is
if "CONDA_PREFIX" in os.environ:
    os.environ["PKG_CONFIG_PATH"] = ":".join(
        [
            os.path.join(os.environ["CONDA_PREFIX"], "lib", "pkgconfig"),
            os.environ.get("PKG_CONFIG_PATH", ""),
        ]
    )
    include_dirs.append(os.path.join(os.environ["CONDA_PREFIX"], "include"))
    library_dirs.append(os.path.join(os.environ["CONDA_PREFIX"], "lib"))

if "VIRTUAL_ENV" in os.environ:
    os.environ["PKG_CONFIG_PATH"] = ":".join(
        [
            os.path.join(os.environ["VIRTUAL_ENV"], "lib", "pkgconfig"),
            os.environ.get("PKG_CONFIG_PATH", ""),
        ]
    )
    include_dirs.append(os.path.join(os.environ["VIRTUAL_ENV"], "include"))
    library_dirs.append(os.path.join(os.environ["VIRTUAL_ENV"], "lib"))
    extra_link_args.append(
        "-Wl,-rpath," + os.path.join(os.environ["VIRTUAL_ENV"], "lib")
    )
include_dirs.append(np.get_include())

have_long_double = not (
    sys.platform.startswith("cygwin")
    or ("macOS" in (_ := platform.platform()) and "arm" in _)
)
fplll_libs = subprocess.check_output(["pkg-config", "fplll", "--libs"])


# Conditional compilation using DEF/IF is deprecated: https://github.com/cython/cython/issues/4310
# NOTE: not using these, just assume all libraries are there
have_qd = b"-lqd" in fplll_libs
have_gmp = b"-lgmp" in fplll_libs
have_mpfr = b"-lmpfr" in fplll_libs
have_long_double = False  # just for simplicity
have_numpy = True  # guaranteed by `build-system.requires`
define_macros = {
    "HAVE_QD": have_qd,
    "HAVE_LONG_DOUBLE": have_long_double,
    "HAVE_NUMPY": have_numpy,
}
assert (
    have_qd and have_gmp and have_mpfr and have_numpy and not have_long_double
), "simplified library assumption"


class Extension(_Extension, object):
    """
    Customized extension with default and shared config
    """

    def __init__(self, name, sources, **kwargs):
        """
        Constructor with shared default config.
        """
        libraries = ["fplll"]
        if have_qd:
            libraries.append("qd")
        if have_gmp:
            libraries.append("gmp")
        if have_mpfr:
            libraries.append("mpfr")

        default = {
            "include_dirs": include_dirs,
            "library_dirs": library_dirs,
            "language": "c++",
            "libraries": libraries,
            "extra_compile_args": ["-std=c++11"] + cxxflags,
            "extra_link_args": extra_link_args,
            # "define_macros": define_macros,
        }
        default.update(kwargs)
        super(Extension, self).__init__(name, sources, **default)


# EXTENSIONS

extensions = [
    Extension("fpylll.gmp.pylong", ["src/fpylll/gmp/pylong.pyx"]),
    Extension(
        "fpylll.fplll.integer_matrix",
        ["src/fpylll/fplll/integer_matrix.pyx"],
    ),
    Extension("fpylll.fplll.gso", ["src/fpylll/fplll/gso.pyx"]),
    Extension("fpylll.fplll.lll", ["src/fpylll/fplll/lll.pyx"]),
    Extension("fpylll.fplll.wrapper", ["src/fpylll/fplll/wrapper.pyx"]),
    Extension("fpylll.fplll.bkz_param", ["src/fpylll/fplll/bkz_param.pyx"]),
    Extension("fpylll.fplll.bkz", ["src/fpylll/fplll/bkz.pyx"]),
    Extension("fpylll.fplll.enumeration", ["src/fpylll/fplll/enumeration.pyx"]),
    Extension("fpylll.fplll.svpcvp", ["src/fpylll/fplll/svpcvp.pyx"]),
    Extension("fpylll.fplll.pruner", ["src/fpylll/fplll/pruner.pyx"]),
    Extension("fpylll.util", ["src/fpylll/util.pyx"]),
    Extension("fpylll.io", ["src/fpylll/io.pyx"]),
    Extension("fpylll.config", ["src/fpylll/config.pyx"]),
    Extension("fpylll.foo", ["src/fpylll/foo.pyx"]),
]
if have_numpy:
    extensions.append(Extension("fpylll.numpy", ["src/fpylll/numpy.pyx"]))

# most config is declarative in `pyproject.toml`
# as suggested by <https://setuptools.pypa.io/en/latest/userguide/quickstart.html#setuppy-discouraged>
setup(
    ext_modules=cythonize(
        extensions,
        build_dir="build",
        compiler_directives={
            "binding": True,
            "embedsignature": True,
            "language_level": 3,
        },
    ),
)
