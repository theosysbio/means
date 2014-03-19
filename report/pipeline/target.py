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

    def dump(self, target):
        with self._file.open('w') as f:
            pickle.dump(target, f, pickle.HIGHEST_PROTOCOL)

    def load(self):
        with self._file.open('r') as f:
            return pickle.load(f)