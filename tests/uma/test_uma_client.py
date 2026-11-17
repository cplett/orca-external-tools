import os
import shutil
import signal
import subprocess
import unittest
from pathlib import Path

from oet.core.test_utilities import (
    OH,
    WATER,
    get_filenames,
    read_hess_file,
    read_result_file,
    run_wrapper,
    wait_for_server,
    write_input_file,
    write_xyz_file,
)

# Path to xtb for the tests involving solvation correction.
# Leave xtb_executable_path empty, if xtb from system path should be called.
xtb_executable_path = ""

# Get the path to the script that should be tested
resolved_uma_client = shutil.which("oet_client")
if resolved_uma_client is None:
    raise RuntimeError(
        "The 'oet_client' script was not found in PATH. "
        "Run the tests with the project's virtual environment activated."
    )
uma_client_path = Path(resolved_uma_client)

resolved_server_script = shutil.which("oet_server")
if resolved_server_script is None:
    raise RuntimeError(
        "The 'oet_server' script was not found in PATH. "
        "Run the tests with the project's virtual environment activated."
    )
uma_server_path = Path(resolved_server_script)

# Get the path to the standalone script for downloading the model files.
resolved_uma_script = shutil.which("oet_uma")
if resolved_uma_script is None:
    raise RuntimeError(
        "The 'goet_xtb' script was not found in PATH. "
        "Run the tests with the project's virtual environment activated."
    )
uma_script_path = Path(resolved_uma_script)

# Default maximum time (in sec) to download the model files if not present
timeout = 600
# Default ID and port of server. Change if needed
ip_port = "127.0.0.1:9000"
# UMA model to use
uma_model = "uma-s-1p1"


def cache_model_files(basemodel: str, param: str = "omol") -> None:
    """
    Wrapper to set up an UMA calculator that downloads the model files into the same cache-directory used for actual oet calculations.

    basemodel: str
        Basemodel used to calculate the test cases
    param: str, default: omol
        Parameter set.
    cache_dir: str, default: DEFAULT_CACHE_DIR
        The cache directory used to store the model data.
    """
    subprocess.run(
        [
            uma_script_path,
            "--download-only",
            "--model",
            basemodel,
            "--task",
            param,
        ],
        check=True,
        timeout=timeout,
    )


def run_uma(
    inputfile: str,
    output_file: str,
    args: list[str] | None = None,
) -> None:
    arguments = args if args else []
    if xtb_executable_path:
        arguments += ["--xtb-solvation-exe", xtb_executable_path]
    run_wrapper(
        inputfile=inputfile,
        script_path=uma_client_path,
        outfile=output_file,
        args=arguments + ["--bind", ip_port, "--model", uma_model],
        timeout=60,
    )


class UmaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Test starting the server
        """
        # Pre-download UMA model files
        print("Checking the model files and downloading them if necessary.")
        try:
            cache_model_files(uma_model)
        except subprocess.TimeoutExpired:
            raise unittest.SkipTest(
                "Loading the model files timed out. Please check your internet connection."
            )
        except subprocess.CalledProcessError:
            raise unittest.SkipTest("Loading the model files failed. ")

        # Start the server
        server_out = Path("server.out").resolve()
        print(f"Starting the server. A detailed server log can be found on file {server_out}")
        with open(server_out, "a") as f:
            cls.server = subprocess.Popen(
                [uma_server_path, "uma", "--bind", ip_port, "--nthreads", "2"],
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        # Wait for the server to be ready.
        wait_for_server(
            process=cls.server,
            ip_port=ip_port,
            timeout=30.0,
        )

    @classmethod
    def tearDownClass(cls):
        """
        Shut the server at the end
        """
        print("Killing the server.")
        os.killpg(os.getpgid(cls.server.pid), signal.SIGTERM)
        cls.server.wait(timeout=10)

    def test_H2O_engrad(self):
        xyz_file, input_file, engrad_out, output_file = get_filenames("H2O_client")

        write_xyz_file(xyz_file, WATER)
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=0,
            multiplicity=1,
            ncores=2,
            do_gradient=1,
        )
        run_uma(input_file, output_file)
        expected_num_atoms = 3
        expected_energy = -76.43352090249
        expected_gradients = [
            -0.007321094162762,
            -0.005420647095889,
            0.003829096443951,
            0.002653303323314,
            0.006170153152198,
            0.001055987784639,
            0.004667790140957,
            -0.0007495055906475,
            -0.004885083995759,
        ]

        try:
            num_atoms, energy, gradients = read_result_file(engrad_out)
        except Exception as e:
            raise FileNotFoundError(
                f"Error wrapper outputfile not found. Check {output_file} for details"
            ) from e

        self.assertEqual(num_atoms, expected_num_atoms)
        self.assertAlmostEqual(energy, expected_energy, places=7)
        for g1, g2 in zip(gradients, expected_gradients):
            self.assertAlmostEqual(g1, g2, places=7)

    def test_H2O_hessian(self):
        xyz_file, input_file, engrad_out, output_file = get_filenames("H2O_hess_client")
        hess_out = "H2O_hess_client.hess"

        write_xyz_file(xyz_file, WATER)
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=0,
            multiplicity=1,
            ncores=2,
            do_gradient=0,
            do_hessian=1,
        )
        run_uma(input_file, output_file)

        expected_num_atoms = 3
        expected_energy = -76.43352090249
        # Reference Hessian taken directly from the UMA calculator (uma-s-1p1, omol) on
        # this geometry, cross-checked against the semi-numerical Hessian of a manual ORCA
        # `! extopt numfreq` run (see tmp-hess-test/orca-uma-numhess.hess): they agree to
        # ~4 significant figures (max abs deviation 1.3e-5 Eh/Bohr^2).
        expected_hessian = [
            0.27651517536245174,
            0.07451102807662664,
            -0.21193898341599532,
            -0.06798138496475879,
            -0.11255250404396952,
            -0.00477389942324221,
            -0.2085337560482475,
            0.03804147842087469,
            0.21671286934481254,
            0.07451104279781756,
            0.48876815196638074,
            0.20788568996798207,
            -0.15126690036889365,
            -0.42305161731570584,
            -0.09907852187236428,
            0.07675586247813972,
            -0.06571649048710217,
            -0.1088071680956178,
            -0.21193902267250442,
            0.20788568996798207,
            0.3315890544887967,
            -0.027269404000211637,
            -0.13598302628820655,
            -0.05740208453903188,
            0.23920841931212058,
            -0.07190266367977553,
            -0.27418695032151025,
            -0.06798139477888605,
            -0.15126690036889365,
            -0.027269418721402542,
            0.0627461123775765,
            0.12226907889067874,
            0.014864759322013447,
            0.005235282401309549,
            0.028997821478214895,
            0.012404659399389095,
            -0.11255249422984225,
            -0.42305157805919674,
            -0.13598302628820655,
            0.12226904944829693,
            0.4365941707945591,
            0.13492240373981684,
            -0.00971656073890127,
            -0.01354261113685097,
            0.0010606162612144228,
            -0.0047738847020513024,
            -0.09907852187236428,
            -0.05740207963196824,
            0.014864751961417995,
            0.1349224135539441,
            0.06803817931928063,
            -0.0100908684861326,
            -0.035843901495707105,
            -0.010636096007014668,
            -0.20853377567650205,
            0.07675586247813972,
            0.23920839968386604,
            0.005235282401309549,
            -0.009716559512135362,
            -0.010090862965686011,
            0.2032984883681289,
            -0.0670393023526214,
            -0.2291175532795198,
            0.03804146369968378,
            -0.06571649048710217,
            -0.07190264895858463,
            0.028997826385278532,
            -0.01354260868331915,
            -0.03584389413511165,
            -0.0670392876314305,
            0.07925908812952814,
            0.1077465455472281,
            0.216712849716558,
            -0.1088071680956178,
            -0.27418695032151025,
            0.012404659399389095,
            0.0010606201715307577,
            -0.01063609232671694,
            -0.22911751402301073,
            0.1077465455472281,
            0.2848230500088227,
        ]

        try:
            num_atoms, energy, gradients = read_result_file(engrad_out)
        except Exception as e:
            raise FileNotFoundError(
                f"Error wrapper outputfile not found. Check {output_file} for details"
            ) from e

        self.assertEqual(num_atoms, expected_num_atoms)
        self.assertAlmostEqual(energy, expected_energy, places=7)
        # No gradient was requested, so none must be written
        self.assertEqual(gradients, [])

        try:
            hessian = read_hess_file(hess_out)
        except Exception as e:
            raise FileNotFoundError(
                f"Error Hessian outputfile not found. Check {output_file} for details"
            ) from e

        self.assertEqual(len(hessian), (3 * expected_num_atoms) ** 2)
        for h1, h2 in zip(hessian, expected_hessian):
            self.assertAlmostEqual(h1, h2, places=5)

        # Check symmetry
        ndim = 3 * expected_num_atoms
        for i in range(ndim):
            for j in range(ndim):
                self.assertAlmostEqual(hessian[i * ndim + j], hessian[j * ndim + i], places=5)

    def test_H2O_solvation_gradient(self):
        xyz_file, input_file, engrad_out, output_file = get_filenames("H2O_solv_grad")

        write_xyz_file(xyz_file, WATER)
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=0,
            multiplicity=1,
            ncores=2,
            do_gradient=1,
        )
        run_uma(input_file, output_file, args=["--xtb-solvent", "water"])
        expected_num_atoms = 3
        expected_energy = -76.43468102541
        expected_gradients = [
            -2.571498316491e-03,
            -1.873541065506e-03,
            1.172834335299e-03,
            2.985381485377e-04,
            3.927222720144e-03,
            2.145703095728e-03,
            2.272959935122e-03,
            -2.053681887468e-03,
            -3.318537780273e-03,
        ]

        try:
            num_atoms, energy, gradients = read_result_file(engrad_out)
        except Exception as e:
            raise FileNotFoundError(
                f"Error wrapper outputfile not found. Check {output_file} for details"
            ) from e

        self.assertEqual(num_atoms, expected_num_atoms)
        self.assertAlmostEqual(energy, expected_energy, places=7)
        for g1, g2 in zip(gradients, expected_gradients):
            self.assertAlmostEqual(g1, g2, places=7)

    def test_H2O_solvation_hessian(self):
        xyz_file, input_file, engrad_out, output_file = get_filenames("H2O_solv_hess")
        hess_out = "H2O_solv_hess.hess"

        write_xyz_file(xyz_file, WATER)
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=0,
            multiplicity=1,
            ncores=2,
            do_gradient=0,
            do_hessian=1,
        )
        run_uma(input_file, output_file, args=["--xtb-solvent", "water"])
        expected_num_atoms = 3
        expected_energy = -76.43468102541

        # Reference Hessian taken directly from the UMA calculator (uma-s-1p1, omol) on
        # this geometry, cross-checked against the semi-numerical Hessian of a manual ORCA
        # `! extopt numfreq` run (see tmp-hess-test/orca-uma-numhess.hess): they agree to
        # ~4 significant figures (max abs deviation 1.3e-5 Eh/Bohr^2).
        expected_hessian = [
            2.7437925513e-01,
            7.3370678574e-02,
            -2.1056555924e-01,
            -6.6410754243e-02,
            -1.1430038966e-01,
            -7.2867296653e-03,
            -2.0796848598e-01,
            4.0929711086e-02,
            2.1785232756e-01,
            7.3370671124e-02,
            4.8656116913e-01,
            2.0769033088e-01,
            -1.4974566108e-01,
            -4.2338531269e-01,
            -1.0072252868e-01,
            7.6374975058e-02,
            -6.3175856448e-02,
            -1.0696781710e-01,
            -2.1056560394e-01,
            2.0769039049e-01,
            3.3016633244e-01,
            -2.7882673295e-02,
            -1.3451075507e-01,
            -5.5961992412e-02,
            2.3844825675e-01,
            -7.3179613066e-02,
            -2.7420436621e-01,
            -6.6410769144e-02,
            -1.4974567598e-01,
            -2.7882682608e-02,
            6.1440811428e-02,
            1.2281310204e-01,
            1.6425153776e-02,
            4.9699609759e-03,
            2.6932570320e-02,
            1.1457529763e-02,
            -1.1430039711e-01,
            -4.2338531269e-01,
            -1.3451074017e-01,
            1.2281307969e-01,
            4.3660521011e-01,
            1.3441022312e-01,
            -8.5126881641e-03,
            -1.3219917083e-02,
            1.0052672802e-04,
            -7.2867259400e-03,
            -1.0072252868e-01,
            -5.5961981236e-02,
            1.6425145394e-02,
            1.3441019331e-01,
            6.6253101719e-02,
            -9.1384180574e-03,
            -3.3687675810e-02,
            -1.0291112201e-02,
            -2.0796851579e-01,
            7.6374975058e-02,
            2.3844825675e-01,
            4.9699623729e-03,
            -8.5126863015e-03,
            -9.1384161948e-03,
            2.0299855072e-01,
            -6.7862292482e-02,
            -2.2930986291e-01,
            4.0929718536e-02,
            -6.3175863899e-02,
            -7.3179620517e-02,
            2.6932579633e-02,
            -1.3219916152e-02,
            -3.3687675810e-02,
            -6.7862292482e-02,
            7.6395783776e-02,
            1.0686730015e-01,
            2.1785232756e-01,
            -1.0696782455e-01,
            -2.7420436621e-01,
            1.1457529763e-02,
            1.0052963840e-04,
            -1.0291115926e-02,
            -2.2930986291e-01,
            1.0686730015e-01,
            2.8449547934e-01,
        ]

        try:
            num_atoms, energy, gradients = read_result_file(engrad_out)
        except Exception as e:
            raise FileNotFoundError(
                f"Error wrapper outputfile not found. Check {output_file} for details"
            ) from e

        self.assertEqual(num_atoms, expected_num_atoms)
        self.assertAlmostEqual(energy, expected_energy, places=7)
        # No gradient was requested, so none must be written
        self.assertEqual(gradients, [])

        try:
            hessian = read_hess_file(hess_out)
        except Exception as e:
            raise FileNotFoundError(
                f"Error Hessian outputfile not found. Check {output_file} for details"
            ) from e

        self.assertEqual(len(hessian), (3 * expected_num_atoms) ** 2)
        for h1, h2 in zip(hessian, expected_hessian):
            self.assertAlmostEqual(h1, h2, places=5)

        # Check symmetry
        ndim = 3 * expected_num_atoms
        for i in range(ndim):
            for j in range(ndim):
                self.assertAlmostEqual(hessian[i * ndim + j], hessian[j * ndim + i], places=5)

    def test_OH_anion_eng_grad(self):
        xyz_file, input_file, engrad_out, output_file = get_filenames("OH_anion_client")
        write_xyz_file(xyz_file, OH)
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=-1,
            multiplicity=1,
            ncores=2,
            do_gradient=1,
        )
        run_uma(input_file, output_file)
        expected_num_atoms = 2
        expected_energy = -75.80575637958
        expected_gradients = [
            -0.001200547791086,
            -0.003864351427183,
            -0.001101008034311,
            0.001200547791086,
            0.003864351427183,
            0.001101008034311,
        ]

        try:
            num_atoms, energy, gradients = read_result_file(engrad_out)
        except Exception as e:
            raise FileNotFoundError(
                f"Error wrapper outputfile not found. Check {output_file} for details"
            ) from e

        self.assertEqual(num_atoms, expected_num_atoms)
        self.assertAlmostEqual(energy, expected_energy, places=7)
        for g1, g2 in zip(gradients, expected_gradients):
            self.assertAlmostEqual(g1, g2, places=7)

    def test_OH_rad_eng_grad(self):
        xyz_file, input_file, engrad_out, output_file = get_filenames("OH_rad_client")
        write_xyz_file(xyz_file, OH)
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=0,
            multiplicity=2,
            ncores=2,
            do_gradient=1,
        )
        run_uma(input_file, output_file)
        expected_num_atoms = 2
        expected_energy = -75.74201333130
        expected_gradients = [
            0.001247821375728,
            0.004016515333205,
            0.001144362613559,
            -0.001247821375728,
            -0.004016515333205,
            -0.001144362613559,
        ]

        try:
            num_atoms, energy, gradients = read_result_file(engrad_out)
        except Exception as e:
            raise FileNotFoundError(
                f"Error wrapper outputfile not found. Check {output_file} for details"
            ) from e

        self.assertEqual(num_atoms, expected_num_atoms)
        self.assertAlmostEqual(energy, expected_energy, places=7)
        for g1, g2 in zip(gradients, expected_gradients):
            self.assertAlmostEqual(g1, g2, places=7)


if __name__ == "__main__":
    unittest.main()
