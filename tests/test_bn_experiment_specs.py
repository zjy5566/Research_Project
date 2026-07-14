import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
SCRIPTS_DIR = os.path.join(PROJECT_DIR, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from run_b_experiments import EXPERIMENTS as B_EXPERIMENTS
from run_n_experiments import EXPERIMENTS as N_EXPERIMENTS


def _matrix(spec):
    return (spec.use_dense, spec.use_tbx, spec.use_sbx, spec.use_patient)


def test_b_experiment_supervision_matrix_is_explicit():
    assert _matrix(B_EXPERIMENTS["b0"]) == (False, False, False, True)
    assert _matrix(B_EXPERIMENTS["b1"]) == (False, True, False, False)
    assert _matrix(B_EXPERIMENTS["b2"]) == (False, False, True, False)
    assert _matrix(B_EXPERIMENTS["b3"]) == (False, True, True, False)
    assert _matrix(B_EXPERIMENTS["b4"]) == (False, True, True, True)


def test_n_experiment_supervision_matrix_is_explicit():
    assert _matrix(N_EXPERIMENTS["n1"]) == (True, False, False, False)
    assert _matrix(N_EXPERIMENTS["n2"]) == (True, True, False, False)
    assert _matrix(N_EXPERIMENTS["n3"]) == (True, False, True, False)
    assert _matrix(N_EXPERIMENTS["n4"]) == (True, True, True, False)
    assert _matrix(N_EXPERIMENTS["n5"]) == (True, True, True, True)


def test_specs_use_best_prior_parameters_and_common_selection():
    b1 = B_EXPERIMENTS["b1"]
    assert b1.lr == 5e-5
    assert b1.pos_weight == 1.0

    b3 = B_EXPERIMENTS["b3"]
    assert b3.use_curriculum
    assert b3.sbx_start == 10

    n4 = N_EXPERIMENTS["n4"]
    assert (n4.dense_weight, n4.tbx_weight, n4.sbx_weight) == (1.0, 0.05, 0.25)
    assert n4.use_curriculum
    assert (n4.tbx_start, n4.sbx_start) == (15, 15)

    all_specs = [*B_EXPERIMENTS.values(), *N_EXPERIMENTS.values()]
    assert all(spec.best_metric == "tbx_roi_auprc" for spec in all_specs)
    assert B_EXPERIMENTS["b4"].patient_weight == 0.05
    assert N_EXPERIMENTS["n5"].patient_weight == 0.05
