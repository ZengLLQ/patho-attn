"""Tests for PathoAttn CLI commands."""

import pytest
from patho_attn.cli import main


def test_cli_help(capsys):
    """Test CLI help."""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "patho-attn" in captured.out


def test_cli_demo(capsys):
    """Test CLI demo execution."""
    code = main(["demo", "--subtype", "1"])
    assert code == 0
    captured = capsys.readouterr()
    assert "PathoAttn" in captured.out
    assert "Predicted Diagnosis" in captured.out
    assert "Attention Diagnostics" in captured.out


def test_cli_eval(capsys):
    """Test CLI cohort eval command."""
    code = main(["eval", "--n-slides", "10"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Cohort Evaluation" in captured.out
    assert "ROC-AUC" in captured.out
