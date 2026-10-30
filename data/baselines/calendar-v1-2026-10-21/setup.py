"""Setuptools hook that packages the root dashboard's external web assets.

The source tree intentionally keeps ``web/`` at the project root because the
development dashboard serves it directly.  Wheels cannot rely on that working
directory, so the build output receives an immutable copy under the Python
package.  The dashboard may prefer the source-tree path during development and
fall back to this packaged path when installed.
"""

from pathlib import Path
from shutil import copytree

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    def run(self):
        super().run()
        source = Path(__file__).parent / "web"
        target = Path(self.build_lib) / "atm_cash_forecast" / "web"
        if source.is_dir():
            copytree(source, target, dirs_exist_ok=True)
            self._web_outputs = [str(path) for path in target.rglob("*") if path.is_file()]
        else:
            self._web_outputs = []

    def get_outputs(self):
        return super().get_outputs() + getattr(self, "_web_outputs", [])


setup(cmdclass={"build_py": build_py})
