"""
Parses yml files that contain a list of "{command: <some_command_to_run>, output_file: <output_file>}" files,
and creates a makefile from them
"""
import argparse
import yaml

def _parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=argparse.FileType('r'))
    parser.add_argument('output', type=argparse.FileType('w'))

    return parser

def main():
    args = _parser().parse_args()

    items = yaml.load(args.input)
    args.input.close()

    all_ = []
    for item in items:
        dependencies = ' '.join(item.get('dependencies', []))
        args.output.write('{0}: {1}\n'.format(item['output'], dependencies))
        args.output.write('\t{0}\n'.format(item['command']))
        all_.append(item['output'])

    args.output.write('all: {0}\n'.format(' '.join(all_)))
    args.output.write('clean:\n')
    for item in all_:
        args.output.write('\trm -f {0}\n'.format(item))

if __name__ == '__main__':
    main()
