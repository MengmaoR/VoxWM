"""User-run real engine GT test: seed42."""
import sys
from test_gt import main

if __name__ == "__main__":
    main(["--case", "seed42", *sys.argv[1:]])
