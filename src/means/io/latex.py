from __future__ import absolute_import, print_function
"""


"""
class LatexPrintableObject(object):

    def _repr_latex_(self):
        raise NotImplementedError

    @property
    def latex(self):
        """
        Returns the latex text that could be printed into .tex document
        :return:
        """
        return self._repr_latex_()

    def output_latex(self, filename_or_file_handle):
        """
        Output the file to a latex document
        :param filename_or_file_handle: filename or already opened file handle to output to
        :return:
        """
        try:
            with open(filename_or_file_handle, 'w') as f:
                f.write(self.latex)
        except TypeError:
            filename_or_file_handle.write(self.latex)
