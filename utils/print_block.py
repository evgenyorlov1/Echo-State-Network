import sys
import os


def block_print():
    """
    Disable print to std.out.
    :return: 
    """
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    """
    Restore print to std.out. 
    :return: 
    """
    sys.stdout = sys.__stdin__