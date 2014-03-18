#/usr/bin/env python2
import os

def main():
    for problem in os.listdir('problems'):
        if not problem.endswith('yml'):
            continue
        all = []
        for parameter in os.listdir('parameters'):
            if not parameter.startswith('parameters'):
                continue

            print '"trajectories/{0}/{1}/": problems/{0} parameters/{1}'.format(problem, parameter)
            print '\tmkdir -p "trajectories/{0}/{1}/"'.format(problem, parameter)
            print '\tpython trajectories_for_problem.py -d "trajectories/{0}/{1}/" ' \
                  'problems/{0} parameters/{1}'.format(problem, parameter)
            all.append('"trajectories/{0}/{1}/"'.format(problem, parameter))

        print 'all: {0}'.format(' '.join(all))


if __name__ == '__main__':
    main()