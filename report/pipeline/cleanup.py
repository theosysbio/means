#!/usr/bin/env python
import argparse
import os

def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('directories', help='Output directories to clean up', nargs='+')

    return parser

def main():
    parser = _parser()
    args = parser.parse_args()

    files = []
    for directory in args.directories:
        dir_files = os.listdir(directory)
        dir_files = filter(lambda x: not x.startswith('.'), dir_files)
        dir_files = map(lambda x: os.path.join(directory, x), dir_files)
        files.extend(dir_files)

    print 'This will remove: '
    for file in files:
        print file

    print '{0} files total'.format(len(files))

    yes_no = raw_input('Continue? y/[n] ')
    if yes_no.lower() == 'y':
        for file in files:
            os.remove(file)
        print 'All done'
    else:
        print 'Negative option selected, aborting'

if __name__ == '__main__':
    main()