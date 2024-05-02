#!/usr/bin/env python3
"""
Common utility functions and classes used by all modules.
"""

import random
import string
from typing import List, Union


# Source: https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def str_generator(size: int = 8, chars: Union[None, List[str]] = None) -> str:
    """
    Generates a string with random sequence of characters.

    Args:

        size (int): Length of string to generate, default = 8

        chars (List[str]): List of characters to use. Default is None, which
        will use a list of ASCII lower-case + list of digits.

    Returns:

        str: The randomly generated string.
    """
    if not chars:
        chars = [*(string.ascii_lowercase + string.digits)]
    return "".join(random.choice(chars) for _ in range(size))


def print_banner(heading: str, print_len: int = 88, return_str: bool = False) -> Union[None, str]:
    """
    Helper routine to construct a pretty-print banner string with heading.

    Args:

        heading (str): Heading to print

        print_len (int): Number of characters to print in a line, default = 88.

        return_str (bool): Return banner as a string, default = False

    Returns:

        None | str: If return_str is true, return constructed banner with
        heading as a string
    """

    # calculate stuff
    heading_len = len(heading)
    left_side_len = (print_len - heading_len - 2) // 2
    right_side_len = print_len - heading_len - left_side_len - 2

    # do the string construction
    s = "=" * print_len
    s += "=" * left_side_len + " " + heading + " " + "=" * right_side_len
    s += "=" * print_len

    if return_str:
        return s

    print(s)

    return None
