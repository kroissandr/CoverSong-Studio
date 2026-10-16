"""python -m coversong — the app (covers + vocal studio). Everything printed is also saved to logs/."""
import os
import sys

if os.environ.get("COVERSONG_CHILD") == "1":
    from .app import main
    main()
else:
    from .cli import tee_main
    sys.exit(tee_main(sys.argv[1:], module="coversong", prefix="app"))
