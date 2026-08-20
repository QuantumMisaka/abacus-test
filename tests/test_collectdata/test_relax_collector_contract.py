from pathlib import Path

import pytest

from abacustest.lib_collectdata.collectdata import RESULT


def _stru(mobility):
    lines = [
        "ATOMIC_SPECIES",
        "H 1.0 H.upf",
        "LATTICE_CONSTANT",
        "1.0",
        "LATTICE_VECTORS",
        "10 0 0",
        "0 10 0",
        "0 0 10",
        "ATOMIC_POSITIONS",
        "Cartesian",
        "H",
        "0.0",
        str(len(mobility)),
    ]
    for index, components in enumerate(mobility):
        lines.append(
            "{0}.0 0.0 0.0 m {1} {2} {3}".format(index, *components)
        )
    return "\n".join(lines) + "\n"


def _force_block(forces):
    rows = ["TOTAL-FORCE (eV/Angstrom)", "--------------------------------"]
    for index, vector in enumerate(forces, 1):
        rows.append("H{0} {1} {2} {3}".format(index, *vector))
    return "\n".join(rows)


def make_result(
    tmp_path,
    *,
    stdout_name="abacus.out",
    stdout=None,
    forces=((100.0, 100.0, 100.0), (0.01, 0.02, 0.03)),
    mobility=((0, 0, 0), (1, 0, 1)),
    calculation="relax",
    relax_new=1,
):
    (tmp_path / "INPUT").write_text(
        "\n".join(
            [
                "INPUT_PARAMETERS",
                f"calculation {calculation}",
                "force_thr_ev 0.02",
                f"relax_new {relax_new}",
                "scf_ene_thr 0.02",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "STRU").write_text(_stru(mobility), encoding="utf-8")
    out_dir = tmp_path / "OUT.ABACUS"
    out_dir.mkdir()
    (out_dir / "INPUT").write_text(
        "INPUT_PARAMETERS\nforce_thr_ev 9.0\n", encoding="utf-8"
    )
    if stdout is None:
        stdout = _force_block(forces)
    (tmp_path / stdout_name).write_text(stdout + "\n", encoding="utf-8")
    return RESULT(path=Path(tmp_path), fmt="abacus")


def test_root_input_is_authoritative_over_out_default(tmp_path):
    result = make_result(tmp_path, stdout_name="abacus.out")

    assert result["INPUT"]["force_thr_ev"] == pytest.approx(0.02)


def test_fallback_largest_gradient_excludes_fixed_atom_and_components(tmp_path):
    result = make_result(tmp_path, stdout_name="custom.stdout")

    assert result["largest_gradient"][-1] == pytest.approx(0.03)


def test_fallback_largest_gradient_is_unknown_when_all_components_are_fixed(tmp_path):
    result = make_result(
        tmp_path,
        stdout_name="custom.stdout",
        mobility=((0, 0, 0), (0, 0, 0)),
    )

    assert result["largest_gradient"] is None


@pytest.mark.parametrize("stdout_name", ["abacus.log", "abacus.out", "custom.stdout"])
def test_stdout_marker_is_discovered_without_fixed_filename(tmp_path, stdout_name):
    result = make_result(
        tmp_path,
        stdout_name=stdout_name,
        stdout="LARGEST GRADIENT = 0.012 eV/Angstrom",
    )

    assert result["largest_gradient"][-1] == pytest.approx(0.012)


def test_old_relax_fallback_requires_energy_difference(tmp_path):
    result = make_result(
        tmp_path,
        stdout_name="abacus.out",
        relax_new=0,
        stdout="largest force 0.01\nenergy difference 0.1\n",
    )

    assert result["relax_converge"] is False


def test_old_relax_without_energy_difference_is_unknown(tmp_path):
    result = make_result(
        tmp_path,
        stdout_name="custom.stdout",
        relax_new=0,
        stdout="largest force 0.01\n",
    )

    assert result["relax_converge"] is None


def test_new_relax_fallback_uses_force_threshold_without_energy_requirement(tmp_path):
    result = make_result(
        tmp_path,
        stdout_name="custom.stdout",
        relax_new=1,
        stdout=_force_block(((0.01, 0.01, 0.01), (0.01, 0.02, 0.01))),
    )

    assert result["relax_converge"] is True


def test_missing_relax_evidence_is_unknown(tmp_path):
    result = make_result(tmp_path, stdout_name="custom.stdout", stdout="no evidence")

    assert result["relax_converge"] is None
