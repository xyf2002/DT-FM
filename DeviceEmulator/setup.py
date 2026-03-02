# install morphling

# classic setup.py

import io
import os
import shutil
import subprocess
import sys
from distutils.command.build import build as _build
from pathlib import Path
from typing import Dict

from setuptools import Command, Extension, find_packages, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools.command.sdist import sdist

try:
    import torch

    torch_available = True
    # The assert is not needed since Github CI does not use GPU server, install cuda library is sufficient
    # assert torch.cuda.is_available() == True
    from torch.utils.cpp_extension import (
        _TORCH_PATH,
        CUDA_HOME,
        EXEC_EXT,
        TORCH_LIB_PATH,
    )

    protoc_path = os.path.join(_TORCH_PATH, "bin", "protoc" + EXEC_EXT)

    print(f"torch version: {torch.__version__}")
except Exception:
    torch_available = False
    print(
        "[WARNING] Unable to import torch, pre-compiling ops will be disabled. "
        "Please visit https://pytorch.org/ to see how to properly install torch on your system."
    )


ROOT_DIR = os.path.dirname(__file__)


def check_nvcc_installed(cuda_home: str) -> None:
    """Check if nvcc (NVIDIA CUDA compiler) is installed."""
    try:
        _ = subprocess.check_output(
            [cuda_home + "/bin/nvcc", "-V"], universal_newlines=True
        )
    except Exception:
        raise RuntimeError(
            "nvcc is not installed or not found in your PATH. "
            "Please ensure that the CUDA toolkit is installed and nvcc is available in your PATH."
        )


assert CUDA_HOME is not None, "CUDA_HOME is not set"
check_nvcc_installed(CUDA_HOME)


def is_ninja_available() -> bool:
    try:
        subprocess.run(["ninja", "--version"], stdout=subprocess.PIPE)
    except FileNotFoundError:
        return False
    return True


def fetch_requirements(path):
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(get_path("README.md"), "r", encoding="utf-8").read()
    else:
        return ""


install_requires = fetch_requirements("requirements.txt")

extras = {}

extras["test"] = [
    "pytest",
    "accelerate>=0.27.2",
    "transformers>=4.37.2",
    "parameterized",
]

sys.path.append(Path.cwd().as_posix())


class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = ".", **kwa) -> None:
        super().__init__(name, sources=[], **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)
        self.target_type = kwa.get("target_type", "shared")


# Adapted from https://github.com/vllm-project/vllm/blob/a1242324c99ff8b1e29981006dfb504da198c7c3/setup.py
class cmake_build_ext(build_ext):
    did_config: Dict[str, bool] = {}

    def configure(self, ext: CMakeExtension) -> None:
        if ext.cmake_lists_dir in cmake_build_ext.did_config:
            return

        cmake_build_ext.did_config[ext.cmake_lists_dir] = True

        default_cfg = "Debug" if self.debug else "Release"
        cfg = os.getenv("CMAKE_BUILD_TYPE", default_cfg)

        outdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name))
        )

        cmake_args = [
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            "-DCMAKE_BUILD_TYPE={}".format(cfg),
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(outdir),
            "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={}".format(outdir),
            "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={}".format(self.build_temp),
        ]

        # verbose = bool(int(os.getenv('VERBOSE', '1')))
        verbose = True
        if verbose:
            cmake_args += ["-DCMAKE_VERBOSE_MAKEFILE=ON"]

        cmake_args += [
            "-DMORPHLING_PYTHON_EXECUTABLE={}".format(sys.executable)
        ]

        if is_ninja_available():
            build_tool = ["-G", "Ninja"]
            cmake_args += [
                "-DCMAKE_JOB_POOL_COMPILE:STRING=compile",
                "-DCMAKE_JOB_POOLS:STRING=compile={}".format(8),
            ]
        else:
            # Default build tool to whatever cmake picks.
            build_tool = []

        if "TEST" in os.environ and os.environ["TEST"] == "1":
            cmake_args.append("-DBUILD_TESTS=ON")
        else:
            cmake_args.append("-DBUILD_TESTS=OFF")

        ccache = shutil.which("ccache")
        if ccache:
            cmake_args += [
                f"-DCMAKE_C_COMPILER_LAUNCHER={ccache}",
                f"-DCMAKE_CXX_COMPILER_LAUNCHER={ccache}",
                f"-DCMAKE_CUDA_COMPILER_LAUNCHER={ccache}",
            ]

        subprocess.check_call(
            ["cmake", ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp,
        )

        if "TEST" in os.environ and os.environ["TEST"] == "1":
            # get folder names under self.build_temp/test/cpp/CmakeFiles
            # and pass them to ninja
            test_folder = os.path.join(self.build_temp, "tests", "cpp")

            with open(
                os.path.join(test_folder, "CTestTestfile.cmake"), "r"
            ) as f:
                content = f.readlines()
                for line in content:
                    if "add_test(" in line:
                        # add_test([=[test_shared_pin_memory]=], get test_shared_pin_memory
                        test_name = (
                            line.strip()
                            .split("add_test([=[")[-1]
                            .split("]=]")[0]
                        )
                        subprocess.check_call(
                            ["ninja", "-C", self.build_temp, test_name]
                        )

    def build_extensions(self) -> None:
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as e:
            raise RuntimeError("Cannot find CMake executable") from e

        # Create build directory if it does not exist.
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Build all the extensions
        for ext in self.extensions:
            self.configure(ext)

            ext_target_name = remove_prefix(ext.name, "morphling.")
            num_jobs = 32

            build_args = [
                "--build",
                ".",
                "--target",
                ext_target_name,
                "-j",
                str(num_jobs),
            ]

            subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)
            print(self.build_temp, ext_target_name)

    def get_ext_filename(self, ext_name):
        """
        Override to manage both shared libraries (.so) and executables.
        """
        for ext in self.extensions:
            if ext.name == ext_name and ext.target_type == "executable":
                # For executables, return the name directly without suffixes
                ext_target_name = ext_name.replace(".", "/")  # TODO: fix this
                return ext_target_name
        # Default behavior for shared libraries
        return super().get_ext_filename(ext_name)


class BuildPackageProtos(Command):
    """Command to generate project *_pb2.py modules from proto files."""

    description = "build grpc protobuf modules"
    user_options = []

    def initialize_options(self):
        self.strict_mode = False

    def finalize_options(self):
        pass

    def _build_package_proto(self, root: str, proto_file: str) -> None:
        command = [
            protoc_path,
            "-I",
            "./",
            f"--python_out={root}",
        ] + [proto_file]
        subprocess.check_call(command)

    def run(self):
        self._build_package_proto(".", "morphling/proto/morphling.proto")


class CustomInstall(install):
    """Custom installation to ensure proto files are compiled and extensions are built before installation."""

    def run(self):
        self.run_command("build_ext")
        self.run_command("build_package_protos")

        super().run()


class CustomBuild(sdist):
    """Custom build command to run build_package_protos."""

    def run(self):
        self.run_command("build_package_protos")
        super().run()


class CustomBdistWheel(bdist_wheel):
    """Custom bdist_wheel command to run build_package_protos."""

    def run(self):
        self.run_command("build_package_protos")
        super().run()


cmdclass = {
    "build_ext": cmake_build_ext,
    "build_package_protos": BuildPackageProtos,
    "install": CustomInstall,
    "sdist": CustomBuild,
    "bdist_wheel": CustomBdistWheel,
}

setup(
    name="morphling",
    version="0.0.1",
    ext_modules=[
        CMakeExtension(name="morphling._C", target_type="shared"),
        CMakeExtension(name="morphling._Msg", target_type="shared"),
        # CMakeExtension(
        #     name="morphling.morphling_server", target_type="executable"
        # ),
        # CMakeExtension(
        #     name="morphling.morphling_worker_server", target_type="executable"
        # ),
    ],
    entry_points={
        "console_scripts": [
            "morphling_emulator=morphling.entrypoint.emulator:main",
            "morphling_device_config=morphling.entrypoint.generate_device_config:main",
            "morphling_device=morphling.entrypoint.run_device:main",
            # "morphling_server=morphling.entrypoint.server:main",
            "morphling_cmd=morphling.entrypoint.cmdline:main",
        ],
    },
    install_requires=install_requires,
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    extras_require=extras,
    packages=find_packages(),
    package_data={
        "morphling": [
            "py.typed",
            "*.so",
            # "morphling_server",
            "morphling_worker_server",
        ],
    },
    include_package_data=True,
    cmdclass=cmdclass,
)
