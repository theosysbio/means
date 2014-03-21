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

        if isinstance(filename_or_file_handle, basestring):
            file_handle = file(filename_or_file_handle, 'w')
            we_opened = True
        else:
            file_handle = filename_or_file_handle
            we_opened = False

        try:
            file_handle.write(self.latex)
        finally:
            if we_opened:
                file_handle.close()
