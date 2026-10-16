"""python -m coversong.studio — the app, opened on the vocal studio page."""
import sys

from ..cli import tee_main

sys.exit(tee_main(["--studio", *sys.argv[1:]], module="coversong", prefix="app"))
