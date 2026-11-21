"""Test isolation: point the portal at throwaway state before any test imports it,
so running tests never wipes the operator's live config_overrides.json."""
import os
import sys
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="cms2-tests-"))
os.environ["CMS_OVERRIDES_FILE"] = str(_TMP / "config_overrides.json")
os.environ["CMS_USERS_FILE"] = str(_TMP / "users.json")
os.environ.setdefault("CMS_SIGNING_KEY", "test-only-signing-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "live_portal_v2"))
