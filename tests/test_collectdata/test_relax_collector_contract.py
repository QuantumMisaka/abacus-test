from pathlib import Path

import pytest

from abacustest.lib_collectdata.collectdata import RESULT


def _stru(mobility, *, style="m"):
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
        flags = "{0} {1} {2}".format(*components)
        if style == "m":
            suffix = f"m {flags}"
        elif style == "bare":
            suffix = flags
        elif style == "bare_velocity":
            suffix = f"{flags} v 0.000000 0.000000 0.000000"
        elif style == "m_velocity":
            suffix = f"m {flags} v 0.000000 0.000000 0.000000"
        elif style == "velocity_m":
            suffix = f"v 0.000000 0.000000 0.000000 m {flags}"
        elif style == "malformed":
            suffix = f"{flags} v invalid 0.000000 0.000000"
        else:
            raise ValueError(style)
        lines.append(f"{index}.0 0.0 0.0 {suffix}")
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
    mobility_style="m",
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
    (tmp_path / "STRU").write_text(_stru(mobility, style=mobility_style), encoding="utf-8")
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


@pytest.mark.parametrize("style", ["bare", "bare_velocity", "m", "m_velocity", "velocity_m"])
def test_force_fallback_honors_mobility_with_optional_velocity(tmp_path, style):
    result = make_result(
        tmp_path,
        stdout_name="abacus.log",
        stdout=_force_block(((0.0, 0.0, -0.5523930502), (0.0, 0.0, -0.0147588701))),
        mobility=((0, 0, 0), (1, 1, 1)),
        mobility_style=style,
    )

    assert result["largest_gradient"][-1] == pytest.approx(0.0147588701)
    assert result["relax_converge"] is True


@pytest.mark.parametrize("gradient, expected", [(0.014759, True), (0.050000, False)])
@pytest.mark.parametrize("unit", ["eV/A", "eV/Angstrom", "eV/Å"])
def test_actual_v3101_gradient_marker_is_authoritative(tmp_path, gradient, expected, unit):
    result = make_result(
        tmp_path,
        stdout_name="abacus.log",
        stdout=f"LARGEST GRAD ({unit})  : {gradient}\n"
        + _force_block(((0.0, 0.0, -0.5523930502), (0.0, 0.0, -0.0147588701))),
        mobility=((0, 0, 0), (1, 1, 1)),
        mobility_style="bare_velocity",
    )

    assert result["largest_gradient"][-1] == pytest.approx(gradient)
    assert result["relax_converge"] is expected


@pytest.mark.parametrize("unit", ["Ry/Bohr", "kbar"])
@pytest.mark.parametrize("with_forces", [False, True])
def test_incompatible_gradient_unit_cannot_claim_force_convergence(tmp_path, unit, with_forces):
    stdout = f"LARGEST GRAD ({unit})  : 0.014759"
    if with_forces:
        stdout += "\n" + _force_block(((0.0, 0.0, -0.5523930502), (0.0, 0.0, -0.05)))
    result = make_result(
        tmp_path,
        stdout_name="abacus.log",
        stdout=stdout,
        mobility=((0, 0, 0), (1, 1, 1)),
        mobility_style="bare_velocity",
    )

    if with_forces:
        assert result["largest_gradient"][-1] == pytest.approx(0.05)
        assert result["relax_converge"] is False
    else:
        assert result["largest_gradient"] is None
        assert result["relax_converge"] is None


def test_malformed_velocity_does_not_erase_force_evidence(tmp_path):
    result = make_result(
        tmp_path,
        stdout_name="abacus.log",
        stdout=_force_block(((0.0, 0.0, -0.5523930502), (0.0, 0.0, -0.0147588701))),
        mobility=((0, 0, 0), (1, 1, 1)),
        mobility_style="malformed",
    )

    assert result["largest_gradient"][-1] == pytest.approx(0.5523930502)
    assert result["relax_converge"] is False


def test_all_fixed_with_velocity_and_no_gradient_marker_stays_unknown(tmp_path):
    result = make_result(
        tmp_path,
        stdout_name="abacus.log",
        stdout=_force_block(((0.0, 0.0, -0.5523930502), (0.0, 0.0, -0.0147588701))),
        mobility=((0, 0, 0), (0, 0, 0)),
        mobility_style="bare_velocity",
    )

    assert result["largest_gradient"] is None
    assert result["relax_converge"] is None


def test_all_mobile_with_velocity_still_checks_force_threshold(tmp_path):
    result = make_result(
        tmp_path,
        stdout_name="abacus.log",
        stdout=_force_block(((0.0, 0.0, -0.022), (0.0, 0.0, -0.014))),
        mobility=((1, 1, 1), (1, 1, 1)),
        mobility_style="bare_velocity",
    )

    assert result["largest_gradient"][-1] == pytest.approx(0.022)
    assert result["relax_converge"] is False
