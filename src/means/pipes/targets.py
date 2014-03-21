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
        self.__cache = object_

    def load(self):
        try:
            return self.__cache
        except AttributeError:
            with self._file.open('r') as f:
                answer = pickle.load(f)

            self.__cache = answer
            return answer

    def get_cache(self):
        try:
            return self.__cache
        except AttributeError:
            return None


class PickleSerialiserWithAdditionalParameters(PickleSerialiser):
    """
    Class similar to PickleSerialiser, however, allows specifying additional parameters apart from the
    original object that would let storing other information, such as runtime as well
    """

    def dump(self, object_, **additional_parameters):
        if 'object' in additional_parameters:
            raise ValueError('Parameter name \'object\' is reserved for the main object that is being stored')

        payload = {'object': object_}
        payload.update(payload, **additional_parameters)

        super(PickleSerialiserWithAdditionalParameters, self).dump(payload)
        # Put things to cache immediately, why not
        self.__cache = payload

    def load(self, parameter='object'):
        all_data = super(PickleSerialiserWithAdditionalParameters, self).load()
        return all_data[parameter]
