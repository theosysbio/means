import sys
import pickle

class ReportUnit(object):
    def __init__(self):
        self.__OUT_FILE_NAME = sys.argv[1]
        self.out_object = None
        self._main()

    def _main(self):
        self.run()
        self._save(self.out_object)

    def _save(self, obj):
        with open(self.__OUT_FILE_NAME, "w") as f:
            pickle.dump(obj, f)

    def run(self):
        raise NotImplementedError


