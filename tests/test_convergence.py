from __future__ import annotations

from cesm_hawc.convergence import extract_l2_native_diagnostics, parse_scipy_convergence


def test_converged_ftol():
    stdout = "some noise\n`ftol` termination condition is satisfied.\nFunction evaluations 42, initial cost..."
    result = parse_scipy_convergence(stdout)
    assert result["converged"] is True
    assert result["termination_reason"] == "ftol"
    assert result["n_function_evaluations"] == 42


def test_not_converged_max_nfev():
    stdout = "The maximum number of function evaluations is exceeded."
    result = parse_scipy_convergence(stdout)
    assert result["converged"] is False
    assert result["termination_reason"] == "max_nfev"


def test_unrecognized_output():
    result = parse_scipy_convergence("nothing recognizable here")
    assert result["converged"] is None
    assert result["termination_reason"] is None
    assert result["n_function_evaluations"] is None


def test_extract_l2_native_diagnostics_none():
    assert extract_l2_native_diagnostics(None) == {"l2_num_iterations": None, "l2_final_cost": None}


def test_extract_l2_native_diagnostics_from_dict_like():
    class FakeL2:
        def __contains__(self, key):
            return key in {"num_iterations", "cost"}

        def __getitem__(self, key):
            class V:
                def __init__(self, v):
                    self.values = v
            return V({"num_iterations": 7, "cost": 0.0123}[key])

    result = extract_l2_native_diagnostics(FakeL2())
    assert result["l2_num_iterations"] == 7
    assert result["l2_final_cost"] == 0.0123
