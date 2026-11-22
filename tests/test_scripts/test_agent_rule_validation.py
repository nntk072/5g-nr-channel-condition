import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        shell=False,
        check=False,
    )


def test_agent_rule_and_profile_validators_pass() -> None:
    for script in ("tools/agent/validate-rules", "tools/agent/validate-profiles"):
        result = _run(script)
        assert result.returncode == 0, result.stdout + result.stderr


def _tmp_agent_tree(tmp_path: Path) -> Path:
    # Fixtures are written only into a temporary copy of the agent tree, never the tracked one.
    tree = tmp_path / "tree"
    for name in (".agents", ".codex", ".claude"):
        shutil.copytree(ROOT / name, tree / name)
    shutil.copytree(ROOT / "tools/agent", tree / "tools/agent")
    shutil.copy2(ROOT / "AGENTS.md", tree / "AGENTS.md")
    return tree


def _run_in(tree: Path, script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(tree / script)],
        cwd=tree,
        text=True,
        capture_output=True,
        shell=False,
        check=False,
    )


def test_rule_validator_rejects_bad_frontmatter_and_broken_link(tmp_path: Path) -> None:
    tree = _tmp_agent_tree(tmp_path)
    (tree / ".agents/rules/INVALID_TEST_RULE.md").write_text(
        "---\ntrigger: invalid_trigger\ndescription: fixture\n---\n[broken](missing-target.md)\n",
        encoding="utf-8",
    )
    result = _run_in(tree, "tools/agent/validate-rules")
    assert result.returncode != 0
    assert "invalid rule trigger" in result.stderr
    assert "broken local link" in result.stderr


def test_rule_validator_rejects_duplicate_role_id(tmp_path: Path) -> None:
    tree = _tmp_agent_tree(tmp_path)
    (tree / ".agents/agents/duplicate_test.md").write_text(
        "# duplicate\n\nCanonical role ID `architect`.\n",
        encoding="utf-8",
    )
    result = _run_in(tree, "tools/agent/validate-rules")
    assert result.returncode != 0
    assert "duplicate canonical role IDs" in result.stderr


def test_profile_validator_rejects_invalid_toml(tmp_path: Path) -> None:
    # Validate a temporary copy of the adapter tree; the tracked adapter is never written.
    tree = _tmp_agent_tree(tmp_path)
    (tree / ".codex/agents/architect.toml").write_text(
        'role_id = "architect"\ninvalid = [\n', encoding="utf-8"
    )
    result = _run_in(tree, "tools/agent/validate-profiles")
    assert result.returncode != 0
    assert "invalid TOML profile" in result.stderr
