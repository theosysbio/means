import luigi
import cPickle as pickle

class PickleSerialiser(luigi.Target):


    def __init__(self, filename):
        self.__filename = filename
        self._file = luigi.File(self.filename)

    @property
    def filename(self):
        return self.__filename

    def exists(self):
        return self._file.exists()

    def dump(self, object_):
        with self._file.open('w') as f:
            pickle.dump(object_, f, pickle.HIGHEST_PROTOCOL)

    def load(self):
        with self._file.open('r') as f:
            return pickle.load(f)


class PickleSerialiserWithAdditionalParameters(PickleSerialiser):
    """
    Class similar to PickleSerialiser, however, allows specifying aditional parameters apart from the
    original object that would let storing other information, such as runtime as well
    """

    def dump(self, object_, **additional_parameters):
        if 'object' in additional_parameters:
            raise ValueError('Parameter name \'object\' is reserved for the main object that is being stored')

        payload = {'object': object_}
        payload.update(payload, **additional_parameters)

        super(PickleSerialiserWithAdditionalParameters, self).dump(payload)

    def load(self, parameter='object'):
        all_data = super(PickleSerialiserWithAdditionalParameters, self).load()
        return all_data[parameter]