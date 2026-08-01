"""Release metadata has one version identity across human-facing files."""

from pathlib import Path
import re

import gridbook
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_citation_and_changelog_match_package_version():
    if not (ROOT / "CITATION.cff").is_file():
        pytest.skip("source-tree release metadata is not shipped in the wheel")
    version = gridbook.__version__
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r'^version:\s*["\']([^"\']+)["\']\s*$', citation,
                      flags=re.MULTILINE)
    assert match is not None, "CITATION.cff has no parseable version field"
    assert match.group(1) == version

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(rf"^## {re.escape(version)}(?:\s|$)", changelog,
                     flags=re.MULTILINE), (
        f"CHANGELOG.md has no release heading for {version}")
