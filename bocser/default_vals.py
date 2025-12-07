from dataclasses import dataclass

from typing import Union

@dataclass
class ConfSearchConfig:
    mol_file_name : str
    spin_multiplicity : int = 1
    charge : int = 0
    orca_exec_command : str = "/opt/orca5/orca"
    num_of_procs : int = 8
    orca_method : str = "lda sto-3g"
    broken_struct_energy : float = 100.
    bond_length_threshold : float = 0.7
    ts : bool = False
    orca_poll_timeout_seconds: int = 3600
    orca_poll_interval_ms: int = 1000
    sbatch_template_name: str = "sbatch_temp"
    rolling_window_size : int = 5
    rolling_std_threshold : float = 0.15
    rolling_mean_threshold : float = 1.
    num_initial_points : int = 3
    max_steps : int = 50
    exp_name : str = "cs"
    load_ensemble : Union[str, None] = None
    acquisition_function : str = "iv"