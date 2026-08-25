#!/usr/bin/env python3

"""Set up test environments and run OET integration tests.

Examples
--------
Run one or more test suite:

    python run_tests.py xtb mopac
    python run_tests.py aimnet2

Use a particular Python interpreter when creating environments:

    python run_tests.py <test_name> --python python3.12

Re-run install.py even when the environment already exists:

    python run_tests.py <test_name> --refresh
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


# Store some paths.
ROOT = Path(__file__).resolve().parent
TESTS_DIR = ROOT / "tests"

# Set some defaults
DEFAULT_VENV_ROOT = ROOT / ".test-venvs"
DEFAULT_BIN_ROOT = ROOT / ".test-bins"

# These backends have additional, potentially incompatible dependencies and
# therefore receive their own virtual environments.
EXTRA_ENVIRONMENTS = {
    "aimnet2": "aimnet2",
    "mace": "mace",
    "mlatom": "mlatom",
    "uma": "uma",
}

# Everything else can use the regular OET installation.
SHARED_ENVIRONMENT = "base"

# Define which tests are currently possible.
# The values correspond to the virtual environment that should be used.
TEST_ENVIRONMENTS = {
    "aimnet2": "aimnet2",
    "g-xtb": SHARED_ENVIRONMENT,
    "mace": "mace",
    "mlatom": "mlatom",
    "mopac": SHARED_ENVIRONMENT,
    "uma": "uma",
    "xtb": SHARED_ENVIRONMENT,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create/reuse OET test environments or test an existing "
            "installation."
        )
    )

    parser.add_argument(
        "targets",
        nargs="+",
        choices=TEST_ENVIRONMENTS,
        help="Test suite(s) to run.",
    )

    parser.add_argument(
        "--python",
        default=sys.executable,
        help=(
            "Python interpreter used to create new managed environments. "
            "Default: the interpreter running this script."
        ),
    )

    parser.add_argument(
        "--venv-root",
        type=Path,
        default=DEFAULT_VENV_ROOT,
        help=(
            "Directory containing managed test virtual environments "
            f"(default: {DEFAULT_VENV_ROOT.relative_to(ROOT)})."
        ),
    )

    parser.add_argument(
        "--bin-root",
        type=Path,
        default=DEFAULT_BIN_ROOT,
        help=(
            "Directory containing managed OET script directories "
            f"(default: {DEFAULT_BIN_ROOT.relative_to(ROOT)})."
        ),
    )

    parser.add_argument(
        "--bin-dir",
        type=Path,
        help=(
            "Use an existing OET bin/script directory instead of a managed "
            "test installation. If --venv-dir is omitted, the virtual "
            "environment is inferred from the oet_* script shebangs."
        ),
    )

    parser.add_argument(
        "--venv-dir",
        type=Path,
        help=(
            "Explicit virtual environment to use with --bin-dir. "
            "If omitted, the environment is inferred from the installed "
            "oet_* scripts."
        ),
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Run install.py again even if the requested managed environment "
            "already exists. Cannot be used with --bin-dir."
        ),
    )

    args = parser.parse_args()

    if args.venv_dir is not None and args.bin_dir is None:
        parser.error("--venv-dir requires --bin-dir")

    if args.bin_dir is not None and args.refresh:
        parser.error("--refresh cannot be used together with --bin-dir")

    return args


def resolve_python(executable: str) -> Path:
    """Resolve the Python executable used to run install.py."""
    path = shutil.which(executable)

    if path is None:
        candidate = Path(executable).expanduser()
        if candidate.exists():
            path = str(candidate.resolve())

    if path is None:
        raise RuntimeError(f"Python interpreter not found: {executable}")

    return Path(path).resolve()


def get_python_interpreter(venv_dir: Path) -> Path:
    """
    Return the Python executable belonging to a virtual environment.

    Parameters
    ----------
    venv_dir: Path
        The path to the virtual environment to get the interpreter from.
    
    Returns
    -------
    Path
        The path of the Python interpreter.
    """

    # Special case of Windows.
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"

    return venv_dir / "bin" / "python"


def environment_is_ready(venv_dir: Path, bin_dir: Path) -> bool:
    """
    Check whether an existing managed test installation looks usable.
    This includes checking whether the Python interpreter exists or not
    and whether there are oet_* scripts in the bin dir.

    Parameters
    ----------
    venv_dir: Path
        The virtual environment.
    bin_dir: Path
        The script dir.
    """
    # Get the Python interpreter and check if it exists
    python = get_python_interpreter(venv_dir)
    if not python.is_file():
        return False

    # Check the bin directory
    if not bin_dir.is_dir():
        return False

    # Check if there is any oet_ script available for testing
    return any(
        path.is_file() and path.name.startswith("oet_")
        for path in bin_dir.iterdir()
    )


def read_shebang_interpreter(script: Path) -> Path | None:
    """
    Return an absolute interpreter path from a script shebang.
    Only absolute interpreter paths are accepted. Shebangs such as
        #!/usr/bin/env python
    are intentionally ignored because they do not identify a specific
    virtual environment.

    Parameters
    ----------
    script: Path
        The path to the scripts.
    
    Returns
    -------
    Path
        The path to the python interpreter inferred from the shebang of the script.
    """
    # Read the shebang.
    try:
        with script.open("r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()
    except (UnicodeDecodeError, OSError):
        return None
    
    # Verify the shebang
    if not first_line.startswith("#!"):
        return None
    shebang = first_line[2:].strip()
    if not shebang:
        return None

    # shlex handles the unlikely case where the interpreter path or shebang
    # contains arguments. We only use the first token as the executable.
    try:
        parts = shlex.split(shebang)
    except ValueError:
        return None
    if not parts:
        return None

    # Get the interpreter
    interpreter = Path(parts[0])
    if not interpreter.is_absolute():
        return None

    return interpreter


def infer_venv_from_bin(bin_dir: Path) -> Path:
    """
    Infer a virtual environment from the installed OET wrapper scripts.

    All readable oet_* scripts with an absolute Python shebang must agree on
    the same virtual environment. If they reference multiple environments,
    inference is considered unsafe and the user must specify --venv-dir.

    Parameters
    ----------
    bin_dir: Path
        The path to the script directory.
        
    Returns
    -------
    Path
        The Path to the virtual environment belonging to the scripts in the bin dir.
    """

    # Resolve the bin path.
    bin_dir = bin_dir.expanduser().resolve()
    if not bin_dir.is_dir():
        raise RuntimeError(
            f"Bin directory does not exist: {bin_dir}"
        )

    # Collet all the scripts found in the bin dir.
    scripts = sorted(
        path
        for path in bin_dir.iterdir()
        if path.is_file() and path.name.startswith("oet_")
    )
    if not scripts:
        raise RuntimeError(
            f"No oet_* scripts found in bin directory: {bin_dir}"
        )

    # Collect all the virtual environments from the shebangs of the scripts.
    candidates: dict[Path, list[str]] = {}
    for script in scripts:
        # First get the interpreter from the shebang
        interpreter = read_shebang_interpreter(script)
        if interpreter is None:
            continue
        # Get the parent directory, which is either `bin` or `scripts`
        # depending on the operating system.
        executable_dir = interpreter.parent
        if executable_dir.name.lower() not in {"bin", "scripts"}:
            continue
        # Get the path to the virtual environments.
        venv_dir = executable_dir.parent.resolve()
        candidates.setdefault(venv_dir, []).append(script.name)

    if not candidates:
        raise RuntimeError(
            "Could not infer a virtual environment from the OET wrappers "
            f"in {bin_dir}. Please provide --venv-dir explicitly."
        )

    # If scripts belonging to different OET installations were found, exit.
    if len(candidates) > 1:
        details = "\n".join(
            f"  {venv}: {', '.join(names)}"
            for venv, names in sorted(
                candidates.items(),
                key=lambda item: str(item[0]),
            )
        )
        raise RuntimeError(
            "The OET scripts reference multiple virtual environments:\n"
            f"{details}\n"
            "Please provide --venv-dir explicitly."
        )

    # Safety check that the python interpreter exists.
    venv_dir = next(iter(candidates))
    python = get_python_interpreter(venv_dir)
    if not python.is_file():
        referenced_scripts = ", ".join(candidates[venv_dir])
        raise RuntimeError(
            "The OET wrappers reference a virtual environment, but its "
            f"Python executable does not exist: {python}\n"
            f"Referenced by: {referenced_scripts}"
        )

    # Inform about the virtual environment used.
    print()
    print("[setup] Inferred virtual environment")
    print(f"[setup] bin:    {bin_dir}")
    print(f"[setup] venv:   {venv_dir}")
    print(f"[setup] Python: {python}")

    return venv_dir


def validate_existing_installation(
    venv_dir: Path,
    bin_dir: Path,
) -> tuple[Path, Path]:
    """
    Validate a user-provided or inferred OET installation.
    
    Parameters
    ----------
    venv_dir: Path
        The path to the virtual environment.
    bin_dir: Path
        The path to the scripts.
    """

    # Resolve the paths
    venv_dir = venv_dir.expanduser().resolve()
    bin_dir = bin_dir.expanduser().resolve()

    # Get the Python interpreter
    python = get_python_interpreter(venv_dir)

    if not venv_dir.is_dir():
        raise RuntimeError(
            f"Virtual environment directory does not exist: {venv_dir}"
        )

    if not python.is_file():
        raise RuntimeError(
            f"No Python executable found in virtual environment: {python}"
        )

    if not bin_dir.is_dir():
        raise RuntimeError(
            f"Bin directory does not exist: {bin_dir}"
        )

    # Collect all the scripts available in the bin directory
    oet_scripts = [
        path
        for path in bin_dir.iterdir()
        if path.is_file() and path.name.startswith("oet_")
    ]

    if not oet_scripts:
        raise RuntimeError(
            f"No OET executables found in bin directory: {bin_dir}"
        )

    # Print the installation found
    print()
    print("[setup] Using existing installation")
    print(f"[setup] venv:   {venv_dir}")
    print(f"[setup] bin:    {bin_dir}")
    print(f"[setup] Python: {python}")

    return venv_dir, bin_dir


def install_environment(
    environment: str,
    *,
    installer_python: Path,
    venv_root: Path,
    bin_root: Path,
    reinstall: bool,
) -> tuple[Path, Path]:
    """
    Create or reuse an OET installation.
    
    Parameters
    ----------
    environment: str
        The name of the environment.
    installer_python: Path
        The path to the Python interpreter used for installation.
    venv_root: Path
        The root of the venv.
    bin_root: Path
        The root of the bin dir.
    reinstall: bool
        Reinstall the oet if already installed.
    """
    # Get the venv and bin dir
    venv_dir = venv_root / environment
    bin_dir = bin_root / environment

    # Check if the environment is ready
    ready = environment_is_ready(venv_dir, bin_dir)

    # If the installation is ready and no reinstallation should be done, return
    if ready and not reinstall:
        print()
        print(f"[setup] Reusing environment '{environment}'")
        print(f"[setup] venv: {venv_dir}")
        print(f"[setup] bin:  {bin_dir}")
        return venv_dir, bin_dir

    # If the installation is ready and reinstallation should be done
    # or if it is not ready, install it.
    if ready:
        print()
        print(f"[setup] Refreshing environment '{environment}'")
    else:
        print()
        print(f"[setup] Setting up environment '{environment}'")

    # Prepare the command
    command = [
        str(installer_python),
        str(ROOT / "install.py"),
        "--venv-dir",
        str(venv_dir),
        "--script-dir",
        str(bin_dir),
    ]
    # Check for extras to be installed.
    extra = EXTRA_ENVIRONMENTS.get(environment)
    if extra is not None:
        command.extend(["--extra", extra])

    print(f"[setup] $ {' '.join(command)}")

    # Run the installation
    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )

    if not environment_is_ready(venv_dir, bin_dir):
        raise RuntimeError(
            f"Installation of environment '{environment}' completed, "
            "but the resulting environment does not look usable."
        )

    return venv_dir, bin_dir


def test_files(target: str) -> list[Path]:
    """Find all test_*.py files belonging to a test target."""
    directory = TESTS_DIR / target

    if not directory.is_dir():
        raise RuntimeError(f"Test directory does not exist: {directory}")

    files = sorted(directory.glob("test_*.py"))

    if not files:
        raise RuntimeError(f"No test files found in {directory}")

    return files


def make_test_environment(
    venv_dir: Path,
    bin_dir: Path,
) -> dict[str, str]:
    """
    Get an environment from the virtual environment that should be tested.
    
    Parameters
    ----------
    venv_dir: Path
        The directory to the virtual environment that should be tested.
    bind_dir: Path
        The directory to the scripts belonging to the venv that should be tested.
    """

    # Initialize a copy of the current environment.
    env = os.environ.copy()

    # The explicit OET bin directory comes first because this is the
    # installation we specifically want to test.
    executable_dirs = [str(bin_dir)]

    if os.name == "nt":
        executable_dirs.append(str(venv_dir / "Scripts"))
    else:
        executable_dirs.append(str(venv_dir / "bin"))

    # Retain the system PATH as well because some tests may depend on
    # external executables such as xtb or mopac.
    old_path = env.get("PATH", "")
    if old_path:
        executable_dirs.append(old_path)

    env["PATH"] = os.pathsep.join(executable_dirs)

    # Make the subprocess behave similarly to an activated virtual
    # environment without depending on shell activation.
    env["VIRTUAL_ENV"] = str(venv_dir)
    env.pop("PYTHONHOME", None)

    return env


def run_test_file(
    test_file: Path,
    *,
    venv_dir: Path,
    bin_dir: Path,
) -> None:
    """
    Run one test script with the respective OET installation.

    Parameters
    ----------
    test_file: Path
        The test file that should be run.
    venv_dir: Path
        The directory of the virtual environment used for running the test.
    bin_dir: Path
        The directory with the scripts used for running the test.
    """

    # Get the python interpreter from the virtual environment
    python = get_python_interpreter(venv_dir)

    if not python.is_file():
        raise RuntimeError(
            f"Python executable missing from virtual environment: {python}"
        )

    # Set the environment up for testing
    env = make_test_environment(venv_dir, bin_dir)

    # Printout of what is tested.
    print()
    print("=" * 78)
    print(f"[test] {test_file.relative_to(ROOT)}")
    print(f"[test] Python: {python}")
    print(f"[test] Bin:    {bin_dir}")
    print("=" * 78)

    # Run the test
    subprocess.run(
        [str(python), str(test_file)],
        cwd=test_file.parent,
        env=env,
        check=True,
    )


def run_target(
    target: str,
    *,
    installer_python: Path,
    venv_root: Path,
    bin_root: Path,
    reinstall: bool,
    installed: dict[str, tuple[Path, Path]],
    external_installation: tuple[Path, Path] | None = None,
) -> list[tuple[Path, subprocess.CalledProcessError]]:
    """
    Set up the required installation and execute one test set.

    Parameters
    ----------
    target: str
        The target test to run.
    installer_python: Path
        The python interpreter for installing the oet if necessary.
    venv_root: Path
        The root dir of the venv for installing the oet if necessary.
    bin_root: Path
        The root dir of the bin for installing the oet if necessary.
    reinstall: bool
        Reinstall the oet if already installed?
    installed: dict[str, tuple[Path, Path]]
        A dictionary with all available installations.
    external_installation: tuple[Path, Path] | None, default: None
        External installation if available.
    
    Returns
    -------
    list[tuple[Path, subprocess.CalledProcessError]]
        A list with potential errors of different tests.
    """
    # If there is an external installation that should be tested, no installation is done.
    if external_installation is not None:
        venv_dir, bin_dir = external_installation
    # Otherwise, install the oet
    else:
        environment = TEST_ENVIRONMENTS[target]

        if environment not in installed:
            installed[environment] = install_environment(
                environment,
                installer_python=installer_python,
                venv_root=venv_root,
                bin_root=bin_root,
                reinstall=reinstall,
            )

        venv_dir, bin_dir = installed[environment]

    # Run the test
    failures: list[tuple[Path, subprocess.CalledProcessError]] = []
    for test_file in test_files(target):
        try:
            run_test_file(
                test_file,
                venv_dir=venv_dir,
                bin_dir=bin_dir,
            )
        except subprocess.CalledProcessError as exc:
            failures.append((test_file, exc))

    return failures


def main() -> int:
    # Parse the arguments
    args = parse_args()

    # Resolve the base directories of the venv and bin
    venv_root = args.venv_root.expanduser().resolve()
    bin_root = args.bin_root.expanduser().resolve()

    # Check which tests should be carried out
    targets = args.targets

    # Check if an existing installation should be checked.
    external_installation: tuple[Path, Path] | None = None
    if args.bin_dir is not None:
        # First the bin dir.
        external_bin_dir = args.bin_dir.expanduser().resolve()
        # Second the venv.
        # Use either a provided venv or try to derive the venv from the scripts.
        if args.venv_dir is not None:
            external_venv_dir = args.venv_dir.expanduser().resolve()
        else:
            external_venv_dir = infer_venv_from_bin(
                external_bin_dir
            )

        # Check if the venv is valid.
        external_installation = validate_existing_installation(
            external_venv_dir,
            external_bin_dir,
        )

        # Get the python interpreter from the provided venv.
        installer_python = get_python_interpreter(external_venv_dir)

    else:
        # If no external venv can be found, get the current python interpreter for installation.
        installer_python = resolve_python(args.python)

    # Print a summary of what will be used
    print()
    print(f"[runner] Repository: {ROOT}")
    print(f"[runner] Targets:    {', '.join(targets)}")

    if external_installation is None:
        print("[runner] Mode:       managed installation")
        print(f"[runner] Python:     {installer_python}")
        print(f"[runner] Venv root:  {venv_root}")
        print(f"[runner] Bin root:   {bin_root}")
    else:
        print("[runner] Mode:       existing installation")

    # Avoid setting up the same shared environment multiple times when running multiple tests.
    # Therefore, keep track about which venvs were already installed.
    installed: dict[str, tuple[Path, Path]] = {}

    passed: list[str] = []
    failed: dict[str, list[tuple[Path, subprocess.CalledProcessError]]] = {}

    for target in targets:
        print()
        print("#" * 78)
        print(f"# Running {target}")
        print("#" * 78)

        failures = run_target(
            target,
            installer_python=installer_python,
            venv_root=venv_root,
            bin_root=bin_root,
            reinstall=args.refresh,
            installed=installed,
            external_installation=external_installation,
        )

        if failures:
            failed[target] = failures
        else:
            passed.append(target)

    # Do a final printout.
    print()
    print("=" * 78)

    # In case anything failed, return with a non-zero return code.
    if failed:
        print("[runner] SOME REQUESTED TESTS FAILED")

        if passed:
            print(f"[runner] Passed: {', '.join(passed)}")

        print("[runner] Failed:")

        for target, failures in failed.items():
            print(f"[runner]   {target}")

            for test_file, exc in failures:
                print(
                    f"[runner]     {test_file.relative_to(ROOT)}: "
                    f"exit code {exc.returncode}"
                )

        print("=" * 78)
        return 1

    # Everything succeeded.
    print("[runner] ALL REQUESTED TESTS PASSED")
    print(f"[runner] Passed: {', '.join(passed)}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())