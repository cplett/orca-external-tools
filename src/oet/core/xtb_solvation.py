"""Optional xTB implicit-solvation correction for ORCA external-tool calculators.

Adds GFNn-xTB solvent-minus-gas energies and gradients to another model's
result. This module uses only the standard library and does not import ML
frameworks or BaseCalc, so it can be imported by both the client and server.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oet.core.base_calc import CalculationData


def extend_parser_by_xtb_solvation_arguments(parser: ArgumentParser) -> None:
    """Register optional flags; `--xtb-solvent` enables the correction."""
    group = parser.add_argument_group("xTB solvation correction")
    group.add_argument(
        "--xtb-solvent",
        default=None,
        dest="xtb_solvent",
        help="Solvent name (e.g. water, thf). Enables xTB solvation correction.",
    )
    group.add_argument(
        "--xtb-solvation-model",
        choices=("alpb", "gbsa", "cosmo"),
        dest="xtb_solvation_model",
        default="cosmo",
        help="Implicit-solvent model (default: cosmo).",
    )
    group.add_argument(
        "--xtb-solvation-method",
        type=str,
        choices=("gfn2", "gfn1", "gfnff"),
        dest="xtb_solvation_method",
        default="gfn2",
        help="Method to compute the solvation contributions.",
    )
    group.add_argument(
        "--xtb-solvation-exe",
        default=None,
        dest="xtb_solvation_executable",
        help="Path or executable name (default: search for xtb).",
    )


def compute_xtb_solvation_correction(
    calc_data: CalculationData, args_parsed: Namespace
) -> tuple[float | None, list[float], list[float]]:
    """Return solvent-minus-gas energy (Eh) and gradient (Eh/Bohr).

    Both xTB calculations use the same molecule and GFN level. Independent
    scratch directories prevent contamination by restart or gradient files.
    """

    # If no solvation contribution can/should be calculated, return.
    if (
        not hasattr(args_parsed, "xtb_solvent")
        or not args_parsed.xtb_solvent
        or args_parsed.xtb_solvent == "gas"
    ):
        return None, [], []

    # xtb solvation is not available for point charges.
    if calc_data.pointcharges is not None:
        raise NotImplementedError(
            "xTB solvation correction with ORCA point-charge embedding is not "
            "implemented. Refusing to mix embedded and unembedded energies."
        )
    # Set the xtb executable
    calc_data = deepcopy(calc_data)
    # Initialize xtb calculator
    from oet.calculator.xtb import XtbCalc

    xtb_calc = XtbCalc()
    # Try to set the default program names first (is usually done when initializing `CalculationData`
    calc_data.set_program_path(xtb_calc.PROGRAM_NAMES)

    # Use the `args_not_parsed` to provide the different possible options for xtb solvation.
    # Use strict. This will cause an error if any unknown CLI flags are used.
    solv_specific_args: list[str] = []
    if args_parsed.xtb_solvation_method:
        solv_specific_args.append(f"--{args_parsed.xtb_solvation_method}")

    # Calculation in gas phase
    basename = calc_data.basename
    output_file = calc_data.output_file
    calc_data.basename = basename + "_xtb_gas"
    calc_data.output_file = output_file.with_stem(output_file.stem + "_xtb_gas")
    gas_results = xtb_calc.calc(
        calc_data=calc_data,
        args_parsed=Namespace(prog=args_parsed.xtb_solvation_executable),
        args_not_parsed=solv_specific_args,
    )
    # Calculation in solvent.
    # Add the solvent model
    solv_specific_args.extend([f"--{args_parsed.xtb_solvation_model}", args_parsed.xtb_solvent])
    calc_data.basename = basename + "_xtb_solvent"
    calc_data.output_file = output_file.with_stem(output_file.stem + "_xtb_solvent")
    solvent_results = xtb_calc.calc(
        calc_data=calc_data,
        args_parsed=Namespace(prog=args_parsed.xtb_solvation_executable),
        args_not_parsed=solv_specific_args,
    )
    # Compute delta G_solv
    delta_e = solvent_results.energy - gas_results.energy
    # Compute gradient contribution
    if calc_data.dograd:
        if (
            len(solvent_results.gradient) != len(gas_results.gradient)
            or len(solvent_results.gradient) != 3 * calc_data.natoms
        ):
            raise ValueError("Gas/solvent xTB gradients have inconsistent lengths")
        delta_g = [solv - gas for solv, gas in zip(solvent_results.gradient, gas_results.gradient)]
    else:
        delta_g = []
    if calc_data.dohess:
        if len(solvent_results.hessian) != len(gas_results.hessian) or len(
            solvent_results.hessian
        ) != (3 * calc_data.natoms) * (3 * calc_data.natoms):
            raise ValueError("Gas/solvent xTB Hessians have inconsistent lengths")
        delta_h = [solv - gas for solv, gas in zip(solvent_results.hessian, gas_results.hessian)]
    else:
        delta_h = []
    print(f"xTB solvation energy correction: {delta_e:+.12f} Eh")
    return delta_e, delta_g, delta_h
