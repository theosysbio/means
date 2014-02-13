#translate expression syntax from the matlab project to the sympy project:
# eg:
#c4*y1-c5*y2,	c5*y2+c6*y3+2*c5*yx3-2*c6*yx2
#c5*y2-c6*y3,	c4*yx5-c5*yx3-c5*y2+c5*yx4-c6*yx3
# would become
#sympy.Matrix([
#            ["c_3*y_0-c_4*y_1", "	c_4*y_1+c_5*y_2+2*c_4*yx3-2*c_5*yx2"],
#            ["c_4*y_1-c_5*y_2", "	c_3*yx5-c_4*yx3-c_4*y_1+c_4*yx4-c_5*yx3"]
#])
# USAGE:
# python matlab2sp.py matlab_exprs > sp_exprs
# where `matlab_exprs` is a textfile with matlabish expressions



import re
import sys 



match_var_idx_re = re.compile('(?P<varname>[yc])(?P<idx>\d+)')

def process_match(match):
	idx = int(match.group("idx")) - 1
	return match.group("varname") +"_{0}".format(idx)

def apply_all(line):
	line = re.sub(r'\^', r' ** ', line)
	line = re.sub(r',', r'", "', line)
	
	line = match_var_idx_re.sub(process_match, line)
	#m = re.match(r'^\["(.*)\"]$', line)
	#if not m:
	
	return '            ["'+line+'"]'


with open(sys.argv[1], "r") as fil:
	sys.stdout.write("sympy.Matrix([\n")
	for i,f in enumerate(fil):
		if i>0:
			sys.stdout.write(",\n")
		line = f.rstrip()
		line = apply_all(line)
		sys.stdout.write(line)
	sys.stdout.write("\n])\n")




