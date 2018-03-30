'''
the framework that are used to execute the code with argparse
'''
import sys


class CommandCall(object):
    '''
    This framework fakes Git that use sub command to call.
    '''

    def __init__(self):
        self.func_dict = {}
        # use dispatch pattern to invoke method with same name

    def add(self, func):
        self.func_dict[func.__name__] = func

    def run(self):
        import argparse as parser_all
        parser = parser_all.ArgumentParser(
            description='Pretends to be git',
            usage='The usage of this script is similar to git, e.g., git [command] [--params]. Please see readme more details. ')
        parser.add_argument('command', help='Subcommand to run')
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        if args.command not in self.func_dict:
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        self.func_dict[args.command]()
