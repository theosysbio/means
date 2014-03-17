import means
import means.examples
import argparse

SUPPORTED_MODELS = {'p53': means.examples.MODEL_P53}

def _one_of_supported_models(str_):
    return SUPPORTED_MODELS[str_]

def _one_of_closure_methods(str_):
    if not str_ in ['gamma', 'log-normal', 'scalar']:
        raise ValueError('Unsupported closure method {0!r}'.format(str_))
    return str_

def _parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Model to use', type=_one_of_supported_models)
    parser.add_argument('max_order', help='Max order to use', type=int)
    parser.add_argument('--closure', help='Closure Method To Use', type=_one_of_closure_methods, default='scalar')
    parser.add_argument('-o', '--output', help='Output file', required=True, type=argparse.FileType('w'))

    return parser

def main():
    parser = _parser()
    args = parser.parse_args()
    problem = means.mea_approximation(args.model,
                                      max_order=args.max_order,
                                      closure=args.closure
                                      )

    problem.to_file(args.output)
    args.output.close()


if __name__ == '__main__':
    main()