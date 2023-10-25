import argparse
import sys
from importlib.metadata import entry_points


def main():
    parser = argparse.ArgumentParser(prefix_chars='+', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '++process', choices=['HYDRO30', 'FD30'], default='HYDRO30',
        help='Select the HyP3 entrypoint to use'  # HyP3 entrypoints are specified in `pyproject.toml`
    )

    args, unknowns = parser.parse_known_args()
    # NOTE: Cast to set because of: https://github.com/pypa/setuptools/issues/3649
    (process_entry_point,) = set(entry_points(group='hyp3', name=args.process))

    sys.argv = [args.process, *unknowns]
    sys.exit(
        process_entry_point.load()()
    )


if __name__ == '__main__':
    main()
