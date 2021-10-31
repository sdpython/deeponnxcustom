# pylint: disable=C0415
"""
@file
@brief Implements command line ``python -m deeponnxcustom <command> <args>``.
"""


def main(args, fLOG=print):
    """
    Implements ``python -m onnxcustom <command> <args>``.

    :param args: command line arguments
    :param fLOG: logging function
    """
    from pyquickhelper.cli import cli_main_helper
    try:
        from . import check
    except ImportError:  # pragma: no cover
        from onnxcustom import check

    fcts = dict(check=check)
    return cli_main_helper(fcts, args=args, fLOG=fLOG)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])  # pragma: no cover
