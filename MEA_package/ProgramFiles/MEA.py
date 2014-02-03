####python MFK_final.py <model> <nMoments> <outputfile>

import sys
from model import parse_model
import ode_problem
from moment_expansion_approximation import MomentExpansionApproximation

def get_args():
    model_ = sys.argv[1]
    numMoments = int(sys.argv[2])
    out_file_name = str(sys.argv[3])
    if numMoments < 2:
        raise ValueError("The number of moments (--nMom) must be greater than one")

    return (model_, numMoments, out_file_name)

if __name__ == "__main__":

    # get and validate command line arguments
    model_filename, n_moments, out_file_name = get_args()

    # parse the input file as a Model object
    model = parse_model(model_filename)

    # set the mea analysis up
    mea = MomentExpansionApproximation(model, n_moments)

    # run mea with the defined parameters
    problem = mea.run()

    # write result in the specified file
    ode_writer = ode_problem.ODEProblemWriter(problem, mea.time_last_run)
    ode_writer.write_to(out_file_name)
    tex_writer = ode_problem.ODEProblemLatexWriter(problem)
    tex_writer.write_to(out_file_name + ".tex")

