"""User-run real engine GT test: test_a."""
import sys
from test_gt import main

if __name__ == "__main__":
    main(["--case", "test_a", *sys.argv[1:]])
