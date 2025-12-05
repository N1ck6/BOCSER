"""
Bayesian Optimization for Conformational Search - Refactored Class-Based Orchestrator

This module provides ConfSearchRunner, a class-based orchestrator that encapsulates
all state and behavior for the conformational search workflow.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Any
import os
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import trieste
import gpflow
from trieste.data import Dataset
from trieste.space import Box
from trieste.models.gpflow.models import GaussianProcessRegression
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition.function import ExpectedImprovement
from rdkit import Chem
from rdkit.Chem import AllChem

from transform_kernel import TransformKernel
from coef_from_grid import pes_tf, pes_tf_grad
from calc import (
    calc_energy,
    load_last_optimized_structure_xyz_block,
    parse_points_from_trj,
    increase_structure_id,
    set_config,
)
from coef_calc import CoefCalculator
from db_connector import LocalConnector
from ensemble_processor import EnsembleProcessor
from evm import ExplorationalVarianceMinimizer
from dbscan import DBSCAN
from default_vals import ConfSearchConfig
from ik_loss import IKLoss
from imp_var_with_ik import ImprovementVarianceWithIK

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

tf.config.run_functions_eagerly(True)


class PotentialFunction:
    """Wrapper for mean function coefficients used in kernel computations."""

    def __init__(self, mean_func_coefs) -> None:
        self.mean_func_coefs = mean_func_coefs

    @tf.function
    def __call__(self, X: tf.Tensor) -> tf.Tensor:
        return tf.stack(
            [
                pes_tf(X[:, dim], *self.mean_func_coefs[dim])
                for dim in range(len(self.mean_func_coefs))
            ],
            axis=1,
        )

    @tf.function
    def grad(self, X: tf.Tensor) -> tf.Tensor:
        return tf.stack(
            [
                pes_tf_grad(X[:, dim], *self.mean_func_coefs[dim])
                for dim in range(len(self.mean_func_coefs))
            ],
            axis=1,
        )


@dataclass
class ConfSearchState:
    """Internal state container for ConfSearchRunner."""

    mol_file_name: Optional[str] = None
    exp_name: str = ""
    structures_path: str = ""
    working_folder: str = ""
    db_file: str = ""
    norm_energy: float = 0.0
    dihedral_ids: list = field(default_factory=list)
    global_degrees: list = field(default_factory=list)
    asked_points: list = field(default_factory=list)
    minima: list = field(default_factory=list)
    model_chk: Optional[Any] = None
    current_minima: float = 1e9
    acq_vals_log: list = field(default_factory=list)
    last_opt_ok: bool = True
    ik_loss: Optional[Any] = None
    ik_loss_dihedrals_idxs: list = field(default_factory=list)
    mean_func_coefs: list = field(default_factory=list)
    search_dim: int = 0
    mol: Optional[Chem.Mol] = None
    config: Optional[ConfSearchConfig] = None


class ConfSearchRunner:
    """
    Orchestrator for Bayesian Optimization-based conformational search.

    Encapsulates all state and workflow logic, eliminating module-level globals.
    """

    def __init__(self, working_folder: str = ".", db_file: Optional[str] = None):
        """
        Initialize the conformational search runner.
        
        Args:
            working_folder: Directory where config, input files are read from
                           and output files are written to. Defaults to current directory.
            db_file: Path to dihedral_logs.db database file. If not provided, defaults to
                    the parent directory of the bocser module (../dihedral_logs.db).
        """
        self.state = ConfSearchState()
        self.state.working_folder = working_folder
        
        # Set database file path
        if db_file is None:
            # Default: one directory up from bocser folder
            bocser_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(bocser_dir)
            db_file = os.path.join(parent_dir, "dihedral_logs.db")
        
        self.state.db_file = db_file
        
        # Ensure working folder exists
        Path(working_folder).mkdir(parents=True, exist_ok=True)

    def _dump_status_hook(self, dumping_value: bool, filename: Optional[str] = None) -> None:
        """Save optimization status to JSON file."""
        if filename is None:
            filename = os.path.join(
                self.state.working_folder,
                self.state.exp_name + "_last_opt_status.json"
            )
        with open(filename, "w") as file:
            json.dump({"LAST_OPT_OK": dumping_value}, file)

    def _calc_point(self, dihedrals: list[float]) -> float:
        """Performs energy calculation for given dihedral angles."""

        if self.state.model_chk:
            print("Checkpoint is not null, calculating previous acq. func. max!")
            dihedrals_tf = tf.constant(dihedrals, dtype=tf.float64)
            if len(dihedrals_tf.shape) == 1:
                dihedrals_tf = tf.reshape(dihedrals_tf, [1, dihedrals_tf.shape[0]])
            print(f"Cur dihedrals_tf: {dihedrals_tf}")
            print(f"Current minima: {self.state.current_minima}")
            mean, variance = self.state.model_chk.predict_f(dihedrals_tf)
            normal = tfp.distributions.Normal(mean, tf.sqrt(variance))
            tau = self.state.current_minima + 3.0
            acq_val = (
                normal.cdf(tau) * (((tau - mean) ** 2) * (1 - normal.cdf(tau)) + variance)
                + tf.sqrt(variance)
                * normal.prob(tau)
                * (tau - mean)
                * (1 - 2 * normal.cdf(tau))
                - variance * (normal.prob(tau) ** 2)
            )
            self.state.acq_vals_log.append(acq_val.numpy().flatten()[0])

        if tf.is_tensor(dihedrals):
            dihedrals = list(dihedrals.numpy())

        self.state.asked_points.append(dihedrals)

        print(f"Point: {dihedrals}")

        # Pre-opt
        print("Optimizing constrained struct")
        en, preopt_status = calc_energy(
            self.state.mol_file_name,
            list(zip(self.state.dihedral_ids, dihedrals)),
            self.state.norm_energy,
            True,
            constrained_opt=True,
            ik_loss=self.state.ik_loss,
        )
        self.state.last_opt_ok = preopt_status
        print(f"Status of preopt: {preopt_status}; LAST_OPT_OK: {self.state.last_opt_ok}")
        if not preopt_status:
            self._dump_status_hook(dumping_value=self.state.last_opt_ok)
            skipped_structure_id = increase_structure_id()
            print(f"Preopt finished with error! Structure with number {skipped_structure_id} will be skipped!")
            return en + np.random.randn()

        print("Optimized!\nLoading xyz from preopt")
        xyz_from_constrained = load_last_optimized_structure_xyz_block(self.state.mol_file_name)
        print("Loaded!\nFull opt")
        en, opt_status = calc_energy(
            self.state.mol_file_name,
            list(zip(self.state.dihedral_ids, dihedrals)),
            self.state.norm_energy,
            True,
            force_xyz_block=xyz_from_constrained,
            ik_loss=self.state.ik_loss,
        )
        self.state.last_opt_ok = opt_status
        print(f"Status of opt: {opt_status}; LAST_OPT_OK: {self.state.last_opt_ok}")
        print(f"Optimized! En = {en}")
        self._dump_status_hook(dumping_value=self.state.last_opt_ok)

        if not opt_status:
            skipped_structure_id = increase_structure_id()
            print(f"Opt finished with error! Structure with number {skipped_structure_id} will be skipped!")

        return en + ((not opt_status) * np.random.randn())

    def _func_objective(self, cur: tf.Tensor) -> tf.Tensor:
        """Defines the objective function for the Bayesian optimization."""
        return tf.map_fn(fn=lambda x: np.array([self._calc_point(x)]), elems=cur)

    def _extract_dofs_values(self, m: Chem.Mol) -> tf.Tensor:
        """Extract dihedral angles from a molecule conformer."""
        return tf.constant(
            [
                [
                    -Chem.rdMolTransforms.GetDihedralRad(
                        m.GetConformer(),
                        *self.state.dihedral_ids[i],
                    )
                    for i in range(len(self.state.dihedral_ids))
                ]
            ],
            dtype=tf.float64,
        )

    def _upd_dataset_from_trj(self, trj_filename: str, dataset: Optional[Dataset]) -> Dataset:
        """Update dataset by parsing trajectory file."""
        print(f"Input dataset is: {dataset}")
        parsed_data, last_point = parse_points_from_trj(
            trj_file_name=trj_filename,
            dihedrals=self.state.dihedral_ids,
            norm_en=self.state.norm_energy,
            save_structs=True,
            structures_path=self.state.structures_path,
            return_minima=True,
        )

        minima_file = os.path.join(
            self.state.working_folder,
            f"{self.state.exp_name}_minima/{len(self.state.minima)}.xyz"
        )
        with open(minima_file, "w") as minima_xyz_writer:
            minima_xyz_writer.write(last_point["xyz_block"])

        self.state.minima.append((last_point["coords"], last_point["rel_en"]))

        print(f"Parsed data: {parsed_data}")
        degrees, energies = zip(*parsed_data)
        print(f"Degrees: {degrees}\nEnergies: {energies}")

        self.state.global_degrees.extend(degrees)

        add_part = Dataset(
            tf.constant(list(degrees), dtype="double"),
            tf.constant(list(energies), dtype="double").reshape(len(energies), 1),
        )

        if not dataset:
            return add_part
        else:
            return dataset + add_part

    def _erase_last_from_dataset(self, dataset: Dataset, n: int = 1) -> Dataset:
        """Remove last n points from dataset."""
        query_points = tf.slice(
            dataset.query_points,
            [0, 0],
            [dataset.query_points.shape[0] - n, dataset.query_points.shape[1]],
        )
        observations = tf.slice(
            dataset.observations,
            [0, 0],
            [dataset.observations.shape[0] - n, dataset.observations.shape[1]],
        )
        return Dataset(query_points, observations)

    def load_config(self, config_path: str) -> None:
        """Load and validate configuration from file.
        
        Args:
            config_path: Path to config file. Can be relative to working_folder or absolute.
        """
        from config_loader import load_config, ConfigError

        # If config_path is relative, look in working_folder
        if not os.path.isabs(config_path):
            config_path = os.path.join(self.state.working_folder, config_path)

        try:
            config = load_config(config_path)
        except FileNotFoundError:
            print(f"No config file {config_path}!\nFinishing!")
            raise
        except ConfigError as e:
            print(f"Config error: {e}\nFinishing!")
            raise
        except Exception:
            print("Something went wrong while loading config!\nFinishing!")
            raise

        self.state.config = config

    def setup(self) -> None:
        """Initialize the search from configuration."""
        if not self.state.config:
            raise RuntimeError("Config not loaded. Call load_config first.")

        config = self.state.config

        print(f"Performing conf. search with config: {config}")

        # Propagate configuration to modules and internal state
        set_config(config)
        
        # Use mol_file_name from config, resolving relative paths to working_folder
        mol_file = config.mol_file_name
        if not os.path.isabs(mol_file):
            mol_file = os.path.join(self.state.working_folder, mol_file)
        self.state.mol_file_name = mol_file
        
        self.state.structures_path = os.path.join(
            self.state.working_folder,
            config.exp_name + "/"
        )
        self.state.exp_name = config.exp_name

        if not os.path.exists(self.state.structures_path):
            os.makedirs(self.state.structures_path)
            minima_path = os.path.join(self.state.working_folder, config.exp_name + "_minima/")
            os.makedirs(minima_path)

        if config.acquisition_function not in {"ei", "evm", "ik"}:
            print(
                f"Acquisition function should be one of the following: 'ei', 'evm', 'ik'; "
                f"got {config.acquisition_function}; Continue with default: 'evm'"
            )
            config.acquisition_function = "evm"

        print("Coef calculator creating")

        self.state.mol = Chem.RemoveHs(Chem.MolFromMolFile(self.state.mol_file_name))
        
        scans_dir = os.path.join(self.state.working_folder, f"{self.state.exp_name}_scans/")
        db_file = os.path.join(self.state.working_folder, "dihedral_logs.db")
        
        coef_calc = CoefCalculator(
            mol=self.state.mol,
            config=config,
            dir_for_inps=scans_dir,
            db_connector=LocalConnector(db_file),
        )
        coef_matrix = coef_calc.coef_matrix()

        print("Coef calculator created!")

        for ids, coefs in coef_matrix:
            self.state.dihedral_ids.append(ids)
            self.state.mean_func_coefs.append(coefs)

        print("Dihedral ids", self.state.dihedral_ids)
        print("Mean func coefs", self.state.mean_func_coefs)

        try:
            dihedral_list_all, ring_atoms_list, ik_loss_dihedrals_idxs = coef_calc.get_ring_dihedrals(
                self.state.mol
            )
            if ik_loss_dihedrals_idxs:
                self.state.ik_loss = IKLoss.from_rdkit(self.state.mol, ring_atoms_list)
                self.state.ik_loss_dihedrals_idxs = ik_loss_dihedrals_idxs
                print(f"IK loss prepared. IK dihedral indices: {ik_loss_dihedrals_idxs}")
            else:
                self.state.ik_loss = None
                self.state.ik_loss_dihedrals_idxs = []
                print("No ring dihedrals detected; IK acquisition will be unavailable.")
        except Exception as e:
            self.state.ik_loss = None
            self.state.ik_loss_dihedrals_idxs = []
            print("Failed to prepare IK loss")
            raise e

        self.state.search_dim = len(self.state.dihedral_ids)
        print("Cur search dim is", self.state.search_dim)

    def _build_model_and_acquisition(self) -> Tuple[Any, Any, Any]:
        """Build GPR model, BO optimizer, and acquisition rule."""
        potential_func = PotentialFunction(self.state.mean_func_coefs)

        kernel = (
            gpflow.kernels.White(0.001)
            + gpflow.kernels.Periodic(
                gpflow.kernels.RBF(
                    variance=0.07,
                    lengthscales=0.005,
                    active_dims=[i for i in range(self.state.search_dim)],
                ),
                period=[2 * np.pi for _ in range(self.state.search_dim)],
            )
            + TransformKernel(
                potential_func,
                gpflow.kernels.RBF(
                    variance=0.12,
                    lengthscales=0.005,
                    active_dims=[i for i in range(self.state.search_dim)],
                ),
            )
        )

        kernel.kernels[1].base_kernel.lengthscales.prior = tfp.distributions.LogNormal(
            loc=tf.constant(0.005, dtype=tf.float64), scale=tf.constant(0.001, dtype=tf.float64)
        )
        kernel.kernels[2].base_kernel.lengthscales.prior = tfp.distributions.LogNormal(
            loc=tf.constant(0.005, dtype=tf.float64), scale=tf.constant(0.001, dtype=tf.float64)
        )

        search_space = Box(
            [0.0 for _ in range(self.state.search_dim)],
            [2 * np.pi for _ in range(self.state.search_dim)],
        )

        # Compute normalizing energy (in kcal/mol)
        self.state.norm_energy, _ = calc_energy(
            self.state.mol_file_name, dihedrals=[], norm_energy=0.0, ik_loss=self.state.ik_loss
        )
        print(f"Norm energy: {self.state.norm_energy}")

        observer = trieste.objectives.utils.mk_observer(self._func_objective)

        return kernel, search_space, observer

    def _initialize_dataset(self, observer: Any) -> Dataset:
        """Build initial dataset from ensemble or random points."""
        config = self.state.config
        dataset = None

        if config.load_ensemble:
            print("Loading init points from given ensemble!")
            dataset = Dataset(
                *EnsembleProcessor(
                    config.load_ensemble,
                    dihedral_idxs=self.state.dihedral_ids,
                ).normalize_energy(self.state.norm_energy).get_tf_data()
            )
            print(f"Init dataset collected!\n{dataset}")
        else:
            for idx in range(config.num_initial_points):
                AllChem.EmbedMolecule(self.state.mol)
                initial_query_points = self._extract_dofs_values(self.state.mol)
                observed_point = observer(initial_query_points)
                if not self.state.last_opt_ok:
                    print(
                        f"Optimization didn't finish well. Continue only with broken_struct_energy "
                        f"in required point: {observed_point}"
                    )
                    dataset = observed_point if not dataset else dataset + observed_point
                else:
                    dataset = self._upd_dataset_from_trj(
                        f"{self.state.mol_file_name[:-4]}_trj.xyz", dataset
                    )

            print(
                f"Initial dataset observed! {config.num_initial_points} minima observed, "
                f"total {dataset.query_points.shape[0]} points has been collected!"
            )

        return dataset

    def _build_rule(self) -> Any:
        """Build acquisition rule based on configuration."""
        config = self.state.config

        rule = None

        match config.acquisition_function:
            case "evm":
                print("Continue with Explorational Variance Minimizer acquisition function!")
                rule = EfficientGlobalOptimization(ExplorationalVarianceMinimizer(threshold=3))
            case "ei":
                print("Continue with ExpectedImprovement acquisition function!")
                rule = EfficientGlobalOptimization(ExpectedImprovement())
            case "ik":
                print("Continue with ImprovementVarianceWithIK acquisition function!")
                if self.state.ik_loss is None or len(self.state.ik_loss_dihedrals_idxs) == 0:
                    print("IK loss is not available; falling back to ExplorationalVarianceMinimizer")
                    rule = EfficientGlobalOptimization(ExplorationalVarianceMinimizer(threshold=3))
                else:
                    rule = EfficientGlobalOptimization(
                        ImprovementVarianceWithIK(
                            threshold=3.0,
                            ik_loss=self.state.ik_loss,
                            ik_loss_idxs=self.state.ik_loss_dihedrals_idxs,
                            ik_loss_weight=1.0,
                        )
                    )
            case _:
                raise ValueError(f"Unknown acquisition function {config.acquisition_function}")

        return rule

    def run(self) -> None:
        """Execute the full Bayesian optimization loop."""
        config = self.state.config

        kernel, search_space, observer = self._build_model_and_acquisition()
        dataset = self._initialize_dataset(observer)

        gpr = gpflow.models.GPR(dataset.astuple(), kernel)
        gpflow.set_trainable(gpr.likelihood, False)
        gpflow.set_trainable(gpr.kernel.kernels[0].variance, False)
        gpflow.set_trainable(gpr.kernel.kernels[1].period, False)
        model = GaussianProcessRegression(gpr, num_kernel_samples=100)

        bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

        print(f"Initial data:\n{dataset}")

        model.optimize(dataset)

        self.state.model_chk = gpflow.utilities.deepcopy(model.model)
        self.state.current_minima = tf.reduce_min(dataset.observations).numpy()

        rule = self._build_rule()

        deepest_minima = []
        early_termination_flag = False

        print(f"MINIMA: {self.state.minima}")

        for step in range(1, config.max_steps + 1):
            print(f"Previous last_opt_ok: {self.state.last_opt_ok}")
            print(f"Step number {step}")

            try:
                result = bo.optimize(1, dataset, model, rule, fit_initial_model=False)
                print(f"Optimization step {step} succeed!")
            except Exception:
                print("Optimization failed")
                print(result.astuple()[1][-1].dataset)

            print(f"After step: {self.state.last_opt_ok}")

            last_opt_status = None
            with open(self.state.exp_name + "_last_opt_status.json", "r") as file:
                last_opt_status = json.load(file)
            print(last_opt_status)

            dataset = result.try_get_final_dataset()
            model = result.try_get_final_model()
            print(f"Last asked point was {self.state.asked_points[-1]}")

            deepest_minima.append(tf.reduce_min(dataset.observations).numpy())

            logs = {
                "acq_vals": self.state.acq_vals_log,
                "deepest_minima": deepest_minima,
                "norm_en": self.state.norm_energy,
            }

            with open(f"{self.state.exp_name}_logs.json", "w") as file:
                json.dump(logs, file)

            print(f"Eta is {rule._acquisition_function._eta}")
            if self.state.last_opt_ok:
                dataset = self._erase_last_from_dataset(dataset, 1)
                dataset = self._upd_dataset_from_trj(
                    f"{self.state.mol_file_name[:-4]}_trj.xyz", dataset
                )
            else:
                print("Last optimization finished with error, skipping trj parsing!")
            model.update(dataset)
            model.optimize(dataset)

            print("Updating model checkpoint!")
            self.state.model_chk = gpflow.utilities.deepcopy(model.model)
            self.state.current_minima = rule._acquisition_function._eta.numpy()[0]
            print("Updated!")

            print(
                f"Step {step} completed! Current dataset is:" + f"\n{dataset}"
            )

            with open(f"{self.state.exp_name}_all_minima.json", "w") as json_minima_writer:
                json.dump(self.state.minima, json_minima_writer)

            if step < config.rolling_window_size:
                continue

            print("Checking termination criterion!")
            print(
                f"Acq vals in window: {logs['acq_vals'][max(0, step - config.rolling_window_size) : step]}"
            )

            rolling_mean = np.mean(
                logs["acq_vals"][max(0, step - config.rolling_window_size) : step]
            )
            rolling_std = np.std(
                logs["acq_vals"][max(0, step - config.rolling_window_size) : step]
            )

            print(f"After step {step}:")
            print(
                f"Current rolling mean of acquisition function maximum is: {rolling_mean}, "
                f"threshold is {config.rolling_mean_threshold}"
            )
            print(
                f"Current rolling std of acquisition function maximum is: {rolling_std}, "
                f"threshold is {config.rolling_std_threshold}"
            )
            if (
                step >= config.rolling_window_size
                and rolling_std < config.rolling_std_threshold
                and rolling_mean < config.rolling_mean_threshold
            ):
                print(f"Termination criterion reached on step {step}! Terminating search!")
                early_termination_flag = True
                break

        if not early_termination_flag:
            print("Max number of steps has been reached!")

        print(f"MINIMA: {self.state.minima}")
        self._save_results(dataset)

    def _save_results(self, dataset: Dataset) -> None:
        """Save final results and ensembles."""
        query_points = dataset.query_points.numpy()
        observations = dataset.observations.numpy()

        dbscan_labels = DBSCAN(
            eps=np.pi / 12,
            min_pts=1,
        ).fit_predict(np.asarray([cur[0] for cur in self.state.minima]))

        res = {int(label): (1e9, -1) for label in np.unique(dbscan_labels)}

        for i in range(len(self.state.minima)):
            cluster_id = dbscan_labels[i]
            if self.state.minima[i][1] < res[cluster_id][0]:
                res[cluster_id] = self.state.minima[i][1], i

        clustering_file = os.path.join(
            self.state.working_folder,
            f"{self.state.exp_name}_clustering_results.json"
        )
        print(
            f"Results of clustering: {res}\n"
            f"There are relative energy and number of structure for each cluster. "
            f"Saved in `{clustering_file}`"
        )
        json.dump(res, open(clustering_file, "w"))

        final_ensemble_file = os.path.join(
            self.state.working_folder,
            f"{self.state.exp_name}_final_ensemble.xyz"
        )
        print(f"Saving final ensemble into `{final_ensemble_file}`")
        ens_xyz_str = ""
        for _, structure_id in res.values():
            cur_xyz = ""
            minima_file = os.path.join(
                self.state.working_folder,
                self.state.exp_name + "_minima",
                f"{structure_id}.xyz"
            )
            with open(minima_file, "r") as cur_xyz_reader:
                cur_xyz = "".join([line for line in cur_xyz_reader])
            ens_xyz_str += cur_xyz + "\n"

        with open(final_ensemble_file, "w") as ens_writer:
            ens_writer.write(ens_xyz_str)

        all_points_file = os.path.join(
            self.state.working_folder,
            f"{self.state.exp_name}_all_points.json"
        )
        print(f"Saving all points at `{all_points_file}`")
        json.dump(
            {"query_points": query_points.tolist(), "observations": observations.tolist()},
            open(all_points_file, "w"),
        )


def main():
    """Entry point for the conformational search orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="bo_confsearch",
        description="Bayesian optimization for conformational search",
    )
    parser.add_argument(
        "--folder",
        default=".",
        help="Working folder for input files and output results (default: current directory)"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Config file name (relative to --folder or absolute path)"
    )

    args = parser.parse_args()

    print(f"Working folder: {args.folder}")
    print(f"Reading config from: {args.config}")

    runner = ConfSearchRunner(working_folder=args.folder)
    runner.load_config(args.config)
    runner.setup()
    runner.run()

    print("Conformational search completed!")


if __name__ == "__main__":
    main()
