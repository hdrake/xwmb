"""xwmb: version information.

The version is derived from the git tag at build time by ``hatch-vcs``, which
writes the resolved value to ``xwmb/_version.py``. That file is generated rather
than checked in: there is deliberately no version string in the source tree that
could drift out of step with the tag it is supposed to describe.

``_version.py`` ships in every built artifact -- wheel, sdist, and editable
install -- so the fallback below only fires when ``xwmb`` is imported straight
from a source checkout that has never been built or installed.
"""

try:
    from xwmb._version import __version__
except ImportError:  # pragma: no cover - un-built source checkout
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
