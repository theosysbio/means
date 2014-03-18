problems.mk: problems.yml
	python makefile-from-yml.py problems.yml problems.mk
-include problems.mk
all: trajectories/%
parameters:
	mkdir parameters
	python generate_parameter_sets.py --per-file=100 -o parameters/parameters.yml
parameters.yml%: parameters
trajectories.mk: parameters problems generate-trajectories-mk.py
	python2 generate-trajectories-mk.py > trajectories.mk
-include trajectories.mk	
clean:
	rm -rf parameters trajectories problems
	rm problems.mk trajectories.mk
