from __future__ import annotations

import logging

from cognitive_impairments.cli import build_parser, run


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    run(build_parser().parse_args())


if __name__ == '__main__':
    main()
