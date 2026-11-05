"""Feasibility model, candidate build and TS checks."""

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from trieste.data import Dataset

import config_manager
from calc import Constraint, change_dihedrals, dihedral_angle, generate_oinp, xyz_coords
from conf_search import ConfSearchRunner
from default_vals import ConfSearchConfig
from feasibility import FeasibilityModel, FeasibilityWeightedAcquisition


# ---------- feasibility model ----------

def test_prior_without_labels():
    fm = FeasibilityModel(n_dihedrals=3, prior=0.8)
    assert fm.probability_at([0.1, 0.2, 0.3]) == pytest.approx(0.8)


def test_one_failure_lowers_p_locally_only():
    fm = FeasibilityModel(n_dihedrals=3, prior=0.8, lengthscale=1.0)
    fm.add([1.0, 2.0, 3.0], False)
    assert fm.probability_at([1.0, 2.0, 3.0]) < 0.3
    assert 0.6 < fm.probability_at([1.0, 2.0, 3.0 + np.pi]) < 0.8  # 180 deg away in one torsion
    assert fm.probability_at([1.0 + np.pi, 2.0 + np.pi, 3.0]) > 0.75  # in two torsions
    assert fm.n_failed == 1


def test_success_raises_p():
    fm = FeasibilityModel(n_dihedrals=2, prior=0.8)
    fm.add([0.5, 0.5], True)
    assert fm.probability_at([0.5, 0.5]) > 0.9


def test_torsions_are_periodic():
    fm = FeasibilityModel(n_dihedrals=1)
    fm.add([0.01], False)
    assert fm.probability_at([2 * np.pi - 0.01]) == pytest.approx(fm.probability_at([0.01]), abs=1e-3)


def test_ts_dimension_uses_its_lengthscale():
    fm = FeasibilityModel(n_dihedrals=1, n_ts_bonds=1, ts_lengthscale=0.5)
    fm.add([1.0, 2.0], False)
    assert fm.probability_at([1.0, 2.0]) < 0.3
    assert fm.probability_at([1.0, 4.0]) > 0.75


def test_bad_points_are_not_recorded():
    fm = FeasibilityModel(n_dihedrals=2)
    assert not fm.add([1.0], False)
    assert not fm.add([1.0, np.nan], False)
    assert len(fm) == 0
    with pytest.raises(ValueError):
        fm.set_data([[1.0]], [1])


class _ConstantFunction:
    _eta = tf.Variable([0.25], dtype=tf.float64)

    def __init__(self, value):
        self.value = value

    def __call__(self, x):
        return tf.fill([tf.shape(x)[0], 1], tf.constant(self.value, tf.float64))


class _ConstantBuilder:
    def __init__(self, value):
        self.value = value

    def prepare_acquisition_function(self, model, dataset=None):
        return _ConstantFunction(self.value)

    def update_acquisition_function(self, function, model, dataset=None):
        return function


@pytest.mark.parametrize("value", [2.0, -2.0])
def test_weighting_prefers_feasible_points_for_both_signs(value):
    fm = FeasibilityModel(n_dihedrals=1)
    fm.add([1.0], False)
    fm.add([4.0], True)
    f = FeasibilityWeightedAcquisition(_ConstantBuilder(value), fm).prepare_acquisition_function(None)
    out = f(tf.constant([[[1.0]], [[4.0]]], dtype=tf.float64)).numpy().ravel()
    assert out.shape == (2,)
    assert out[1] > out[0]
    assert f._eta.numpy()[0] == pytest.approx(0.25)


def test_new_labels_reach_a_traced_function():
    fm = FeasibilityModel(n_dihedrals=1)
    f = tf.function(lambda x: fm.probability(x))
    before = float(f(tf.constant([[1.0]], tf.float64))[0, 0])
    fm.add([1.0], False)
    assert float(f(tf.constant([[1.0]], tf.float64))[0, 0]) < before


def test_ego_acquires_away_from_failed_region():
    """End-to-end with trieste: the weighted EVM picks a point and avoids a failed cluster."""
    import gpflow
    from trieste.acquisition.rule import EfficientGlobalOptimization
    from trieste.models.gpflow.models import GaussianProcessRegression
    from trieste.space import Box
    from evm import ExplorationalVarianceMinimizer

    x = np.array([[0.5, 0.5], [5.5, 5.5], [3.0, 0.5]])
    y = np.array([[1.0], [2.0], [1.5]])
    gpr = gpflow.models.GPR((x, y), gpflow.kernels.Periodic(gpflow.kernels.RBF(lengthscales=[0.7, 0.7]), period=[2 * np.pi] * 2),
                            mean_function=gpflow.mean_functions.Constant(1.5))
    model = GaussianProcessRegression(gpr)
    fm = FeasibilityModel(n_dihedrals=2)
    rule = EfficientGlobalOptimization(FeasibilityWeightedAcquisition(ExplorationalVarianceMinimizer(threshold=3), fm))
    space = Box([0.0, 0.0], [2 * np.pi, 2 * np.pi])
    first = rule.acquire_single(space, model, dataset=Dataset(tf.constant(x), tf.constant(y))).numpy()[0]
    for dx in (-0.2, 0.0, 0.2):
        for dy in (-0.2, 0.0, 0.2):
            fm.add([first[0] + dx, first[1] + dy], False)
    second = rule.acquire_single(space, model, dataset=Dataset(tf.constant(x), tf.constant(y))).numpy()[0]
    d = np.abs(second - first) % (2 * np.pi)
    assert np.max(np.minimum(d, 2 * np.pi - d)) > 0.3


# ---------- runner pieces ----------

def _runner(n_dih=2, feasibility=True):
    runner = ConfSearchRunner()
    runner.state.config = ConfSearchConfig(mol_file_name="x.mol")
    runner.state.dihedral_ids = [[0, 1, 2, 3]] * n_dih
    runner.state.search_dim = n_dih
    if feasibility:
        runner.state.feasibility = FeasibilityModel(n_dihedrals=n_dih)
    return runner


def test_record_feasibility_labels():
    r = _runner()
    r.state.last_eval_valid = False
    r._record_feasibility([1.0, 1.0])
    assert r.state.feasibility.labels == [0]

    r.state.last_eval_valid, r.state.last_built_point = True, [1.0, 1.0 + 0.1]
    r._record_feasibility([1.0, 1.0])
    assert r.state.feasibility.labels[-1] == 1

    # A label always belongs to the asked point (far builds are rejected before ORCA)
    r.state.last_built_point = [1.0, 2.5]
    r._record_feasibility([1.0, 1.0])
    assert r.state.feasibility.points[-1] == [1.0, 1.0]
    assert r.state.feasibility.labels[-1] == 1


@pytest.mark.parametrize("reason, labelled", [("no_out_file", False), ("parse_error", False),
                                              ("max_iterations", True), ("clash", True), (None, True)])
def test_technical_failures_are_not_labelled(reason, labelled):
    r = _runner()
    r.state.last_eval_valid, r.state.last_failure_reason = False, reason
    r._record_feasibility([1.0, 1.0])
    assert len(r.state.feasibility) == (1 if labelled else 0)


def test_default_lengthscale_separates_rotamers():
    fm = FeasibilityModel(n_dihedrals=1)  # default 0.5
    fm.add([0.0], False)
    assert fm.probability_at([0.0]) < 0.3
    assert fm.probability_at([np.radians(20)]) < 0.4
    assert fm.probability_at([np.radians(120)]) > 0.79  # the next staggered position is untouched


def test_mmff_reference_and_limit():
    r = _runner()
    r.state.config = ConfSearchConfig(mol_file_name="x.mol", mmff_energy_window=30.0)
    assert r._mmff_energy_limit() is None  # nothing known yet: no filter
    r._note_mmff_energy(12.0)
    r._note_mmff_energy(20.0)
    r._note_mmff_energy(None)
    assert r._mmff_energy_limit() == pytest.approx(42.0)
    r.state.config = ConfSearchConfig(mol_file_name="x.mol", mmff_energy_window=None)
    assert r._mmff_energy_limit() is None


def test_observation_moves_to_built_point():
    r = _runner()
    ds = Dataset(tf.constant([[0.0, 0.0], [1.0, 1.0]], tf.float64), tf.constant([[1.0], [2.0]], tf.float64))
    r.state.last_eval_valid, r.state.last_built_point = True, [1.0, 1.5]
    moved = r._move_last_to_built_point(ds)
    np.testing.assert_allclose(moved.query_points.numpy()[-1], [1.0, 1.5])
    np.testing.assert_allclose(moved.query_points.numpy()[0], [0.0, 0.0])
    r.state.last_eval_valid = False
    assert r._move_last_to_built_point(ds) is ds


def test_broken_observations_leave_the_gp():
    r = _runner()
    ds = Dataset(tf.constant([[0.0, 0.0], [1.0, 1.0]], tf.float64), tf.constant([[1.0], [100.3]], tf.float64))
    kept = r._drop_invalid_observations(ds)
    assert kept.observations.numpy().ravel().tolist() == [1.0]
    assert _runner(feasibility=False)._drop_invalid_observations(ds) is ds


def test_demoted_ring_axes_stay_free():
    r = _runner(n_dih=10)
    r.state.fixed_dihedrals = []
    r.state.mean_func_coefs = [(0,) * 7] * 10
    windows = [[k, k + 1, k + 2, k + 3] for k in range(10)]
    r.state.dihedral_ids = windows
    cycle_idx = [list(range(10))]
    r._apply_ring_dof_cap([windows], cycle_idx)
    assert len(r.state.dihedral_ids) == 4          # 10 flexible positions - 6
    assert len(r.state.free_ring_dihedrals) == 6
    assert r.state.fixed_dihedrals == []           # not frozen any more
    assert cycle_idx[0].count(-4) == 6


def test_acquisition_ik_loss_skips_rings_with_free_torsions():
    r = _runner()
    r.state.ik_loss = object()
    r.state.ik_loss_dihedrals_idxs = [[0, 1, -4, -4, -2]]
    r._prepare_acquisition_ik_loss()
    assert r.state.ik_acq_loss is None
    r.state.ik_loss_dihedrals_idxs = [[0, 1, -2]]
    r._prepare_acquisition_ik_loss()
    assert r.state.ik_acq_loss is r.state.ik_loss


def test_left_ts_region():
    r = _runner()
    r.state.mol_file_name = "x.mol"
    r.state.ts_bond_windows = {(0, 1): (2.0, 3.0)}
    xyz = "C 0 0 0\nC {} 0 0\n"
    import conf_search
    orig = conf_search.with_h_canonical_to_raw1
    conf_search.with_h_canonical_to_raw1 = lambda a, _: a + 1
    try:
        assert r._left_ts_region(xyz.format(2.5)) == (None, None)
        assert r._left_ts_region(xyz.format(1.55))[0] == "left_ts_region"
        assert r._left_ts_region(xyz.format(3.4))[0] == "left_ts_region"
    finally:
        conf_search.with_h_canonical_to_raw1 = orig


def test_nearest_template():
    r = _runner()
    r.state.templates = [(np.array([0.0, 0.0]), [[0, 0, 0]]), (np.array([3.0, 3.0]), [[1, 1, 1]])]
    assert r._nearest_template([2.8, 3.2]) == [[1, 1, 1]]
    assert r._nearest_template([6.2, 0.1]) == [[0, 0, 0]]  # periodic
    r.state.config.build_from_nearest_conformer = False
    assert r._nearest_template([2.8, 3.2]) is None


# ---------- calc: build and ORCA input ----------

@pytest.fixture
def butane_mol(tmp_path):
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(mol)
    path = tmp_path / "butane.mol"
    Chem.MolToMolFile(mol, str(path))
    return path


def _set_config(**kw):
    cfg = ConfSearchConfig(mol_file_name="x.mol", **kw)
    config_manager.set_config(cfg)
    return cfg


def test_ik_build_reaches_the_asked_torsion(butane_mol):
    _set_config(acquisition_function="ik")
    target = 1.2  # BO convention (-RDKit), radians
    dihedrals = [((0, 1, 2, 3), target)]
    xyz = change_dihedrals(str(butane_mol), dihedrals)
    coords = xyz_coords(xyz)
    built = dihedral_angle(*(coords[i] for i in (0, 1, 2, 3)))
    assert abs((built - target + np.pi) % (2 * np.pi) - np.pi) < np.radians(3)
    assert dihedrals[0][1] == pytest.approx(built)


def test_ik_build_holds_ts_bond(butane_mol):
    _set_config(acquisition_function="ik")
    xyz = change_dihedrals(str(butane_mol), [((0, 1, 2, 3), 3.0)], ts_bonds=[(1, 2)], ts_bond_targets=[((1, 2), 2.6)])
    coords = xyz_coords(xyz)
    assert np.linalg.norm(coords[1] - coords[2]) == pytest.approx(2.6, abs=0.05)


def test_build_from_start_coords(butane_mol):
    _set_config(acquisition_function="evm")
    mol = Chem.MolFromMolFile(str(butane_mol), removeHs=False)
    start = (mol.GetConformer().GetPositions() + 10.0).tolist()
    xyz = change_dihedrals(str(butane_mol), [((0, 1, 2, 3), 2.0)], start_coords=start)
    assert xyz_coords(xyz)[:, 0].mean() > 5.0


def test_orca_input_ts_mode_and_current_constraints(tmp_path):
    _set_config(ts=True)
    inp = tmp_path / "a.inp"
    generate_oinp(
        "C 0 0 0\n", [], str(inp), 1, "M062X def2-SVP", 0, 1,
        hard_constraints=[Constraint("dihedral", (0, 1, 2, 3), 61.5, True), Constraint("angle", (0, 1, 2), 109.0)],
        ts_mode_bond=(12, 13),
    )
    text = inp.read_text()
    assert "{ D 0 1 2 3 C }" in text
    assert "{ A 0 1 2 109.0 C }" in text
    assert "TS_Mode {B 12 13} end" in text
    assert "Calc_Hess true" in text


def test_preopt_input_has_no_ts_mode(tmp_path):
    _set_config(ts=True)
    inp = tmp_path / "p.inp"
    generate_oinp("C 0 0 0\n", [], str(inp), 1, "M062X def2-SVP", 0, 1, constrained_opt=True, ts_mode_bond=(1, 2))
    assert "TS_Mode" not in inp.read_text()


def test_config_validation():
    with pytest.raises(ValueError):
        ConfSearchConfig(mol_file_name="x.mol", feasibility_prior=1.0)
    with pytest.raises(ValueError):
        ConfSearchConfig(mol_file_name="x.mol", ts_mode_bond=[1, 2, 3])
    assert ConfSearchConfig(mol_file_name="x.mol", ts_mode_bond=["3", "4"]).ts_mode_bond == [3, 4]
    with pytest.raises(ValueError):
        ConfSearchConfig(mol_file_name="x.mol", mmff_energy_window=0)
    with pytest.raises(ValueError):
        ConfSearchConfig(mol_file_name="x.mol", candidate_batch_size=0)


# ---------- MMFF build without fallback, MMFF energy window ----------

def _ik_calc(mol_path, tmp_path, **kw):
    from calc import calc_energy
    failure = {}
    en, ok = calc_energy(
        str(mol_path), [((0, 1, 2, 3), 1.2)], 0.0, constrained_opt=True,
        broken_structs_dir=str(tmp_path / "broken"), failure_info=failure, **kw,
    )
    return ok, failure


def test_build_far_from_asked_is_rejected(butane_mol, tmp_path):
    _set_config(acquisition_function="ik")
    ok, failure = _ik_calc(butane_mol, tmp_path, max_build_deviation=1e-6)
    assert not ok and failure == {"stage": "geometry", "reason": "mmff_off_target"}
    assert any("mmff_off_target" in p.name for p in (tmp_path / "broken").iterdir())


def test_mmff_not_converged_is_rejected(butane_mol, tmp_path, monkeypatch):
    _set_config(acquisition_function="ik")
    import calc
    original = calc.change_dihedrals

    def not_converged(*args, build_status=None, **kw):
        xyz = original(*args, **kw)
        build_status["mmff_not_converged"] = True
        return xyz

    monkeypatch.setattr(calc, "change_dihedrals", not_converged)
    ok, failure = _ik_calc(butane_mol, tmp_path)
    assert not ok and failure["reason"] == "mmff_not_converged"


def test_build_exception_is_rejected(butane_mol, tmp_path, monkeypatch):
    _set_config(acquisition_function="ik")
    import calc

    def boom(*args, **kw):
        raise RuntimeError("no MMFF parameters")

    monkeypatch.setattr(calc, "change_dihedrals", boom)
    ok, failure = _ik_calc(butane_mol, tmp_path)
    assert not ok and failure == {"stage": "geometry", "reason": "build_error"}


def test_mmff_energy_window_rejects(butane_mol, tmp_path):
    _set_config(acquisition_function="ik", mmff_energy_window=30.0)
    info = {}
    ok, failure = _ik_calc(butane_mol, tmp_path, mmff_energy_limit=-1e6, build_info=info)
    assert not ok and failure["reason"] == "mmff_energy_high"
    assert info["mmff_energy"] is not None


def test_mmff_energy_leaves_out_the_ts_bond(butane_mol):
    from calc import mmff_energy
    _set_config(acquisition_function="ik")
    mol = Chem.MolFromMolFile(str(butane_mol), removeHs=False)
    energies = {}
    for length in (2.0, 2.4):
        m = Chem.Mol(mol)
        rdMolTransforms.SetBondLength(m.GetConformer(), 0, 1, length)
        block = "\n".join(Chem.MolToXYZBlock(m).split("\n")[2:])
        energies[length] = (mmff_energy(block, str(butane_mol)), mmff_energy(block, str(butane_mol), ts_bonds=[(0, 1)]))
    full = energies[2.4][0] - energies[2.0][0]
    without_ts = energies[2.4][1] - energies[2.0][1]
    assert full > 20.0  # the stretched C-C term alone
    assert abs(without_ts) < 0.25 * full


# ---------- several candidates from one acquisition search ----------

def test_batch_optimizer_keeps_distinct_maxima():
    from trieste.space import Box
    from candidate_batch import BatchRecordingOptimizer

    def acq(x):  # two separate maxima, at 1.0 and 4.0 along both axes
        x = tf.squeeze(x, -2)
        bump = lambda c: tf.exp(-tf.reduce_sum(tf.square(x - c), axis=-1, keepdims=True))
        return 2.0 * bump(1.0) + bump(4.0)

    opt = BatchRecordingOptimizer(batch_size=3, n_dihedrals=2)
    best = opt(Box([0.0, 0.0], [2 * np.pi, 2 * np.pi]), acq).numpy()[0]
    np.testing.assert_allclose(best, [1.0, 1.0], atol=1e-2)
    assert len(opt.candidates) >= 2
    nxt = opt.pop_next(acq, tried=[best])
    np.testing.assert_allclose(nxt, [4.0, 4.0], atol=1e-2)
    assert opt.pop_next(acq, tried=[best, nxt]) is None


# ---------- IK loss: TS bond with a length range ----------

class _Conf:
    def __init__(self, xyz):
        self.x = xyz

    def l(self, i, j):
        return float(np.linalg.norm(self.x[i - 1] - self.x[j - 1]))

    def v(self, i, j, k):
        u, w = self.x[i - 1] - self.x[j - 1], self.x[k - 1] - self.x[j - 1]
        return float(np.degrees(np.arccos(np.dot(u, w) / np.linalg.norm(u) / np.linalg.norm(w))))


def _ik_for(mol, rings, monkeypatch, **kw):
    """IKLoss.from_rdkit with ringo's two geometry calls computed here (ringo-ik is optional)."""
    import contextlib, types
    import ik_loss

    class Pool:
        def include_from_file(self, name):
            pass

        def __getitem__(self, i):
            return _Conf(Chem.AddHs(mol, addCoords=True).GetConformer().GetPositions())

    @contextlib.contextmanager
    def no_file(suffix="", delete=True):
        yield types.SimpleNamespace(name=str(Path(ik_tmp) / ("m" + suffix)))

    import tempfile as _tf
    ik_tmp = _tf.mkdtemp()
    monkeypatch.setattr(ik_loss, "ringo", types.SimpleNamespace(Confpool=Pool))
    monkeypatch.setattr(ik_loss, "tempfile", types.SimpleNamespace(NamedTemporaryFile=no_file))
    return ik_loss.IKLoss.from_rdkit(mol, rings, **kw)


def _ring_path(mol, ring):
    path = [ring[0]]
    while len(path) < len(ring):
        path.append(next(n.GetIdx() for n in mol.GetAtomWithIdx(path[-1]).GetNeighbors()
                         if n.GetIdx() in ring and n.GetIdx() not in path))
    return path


def test_ik_open_ts_bond_distance_and_range(monkeypatch):
    mol = Chem.AddHs(Chem.MolFromSmiles("C1CCCCCCC1"))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    ring = _ring_path(mol, list(mol.GetRingInfo().AtomRings()[0]))
    a, b = ring[2], ring[3]
    true = mol.GetConformer().GetAtomPosition(a).Distance(mol.GetConformer().GetAtomPosition(b))
    ref = [tf.constant(np.full((1, len(ring)), np.nan))]

    closed = _ik_for(mol, [ring], monkeypatch)
    assert float(tf.reduce_sum(closed(ref))) < 1e-8

    inside = _ik_for(mol, [ring], monkeypatch, ts_bond_ranges={frozenset((a, b)): (1.0, 5.0)})
    d = float(inside.open_chain_distance(0, inside.build_T_matrices(ref)[0])[0])
    assert d == pytest.approx(true, abs=1e-6)
    assert float(tf.reduce_sum(inside(ref))) == pytest.approx(0.0)

    outside = _ik_for(mol, [ring], monkeypatch, ts_bond_ranges={frozenset((a, b)): (true + 0.3, 5.0)})
    assert float(tf.reduce_sum(outside(ref))) == pytest.approx(0.09, abs=1e-6)


# ---------- scans of ring torsions on open fragments ----------

def test_ring_torsions_get_open_fragments(tmp_path):
    from coef_calc import CoefCalculator
    from db_connector import LocalConnector
    mol = Chem.AddHs(Chem.MolFromSmiles("CC1CCCCCC(=O)N1"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    mol = Chem.RemoveHs(mol)
    cfg = _set_config(acquisition_function="ik")
    cc = CoefCalculator(mol, cfg, dir_for_inps=str(tmp_path), db_connector=LocalConnector(str(tmp_path / "d.db")),
                        submit_scans=False)
    _, rings, _ = cc.get_ring_dihedrals(mol)
    ring = set(rings[0])
    assert cc.generate_scan_inps_from_mol() and not cc.flat_ring_smiles
    by_index = {i: smi for smi, i in cc.unique_frags.items()}
    assert all(Chem.MolFromSmiles(smi).GetRingInfo().NumRings() == 0 for smi in by_index.values())
    ring_keys = [k for k in cc.frags if set(k) <= ring]
    assert len(ring_keys) >= 5
    for key in ring_keys:  # keyed on ring atoms (IK), scanned on a mapped open fragment
        assert cc._mapped_torsion(Chem.MolFromSmiles(by_index[cc.frags[key]])) is not None
