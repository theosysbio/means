import unittest
from means.io.serialise import dump, load
from means.examples.sample_models import MODEL_P53, MODEL_MICHAELIS_MENTEN, MODEL_LOTKA_VOLTERRA, \
                                         MODEL_HES1, MODEL_DIMERISATION


class TestSerialisation(unittest.TestCase):

    def _roundtrip(self, object_):
        self.assertEquals(load(dump(object_)), object_)

    def test_model_serialisation_works(self):
        """
        Given a model object, the serialisation routine should be able to dump that model and recover it when
        the dumped data is loaded
        :return:
        """

        self._roundtrip(MODEL_P53)
        self._roundtrip(MODEL_MICHAELIS_MENTEN)
        self._roundtrip(MODEL_LOTKA_VOLTERRA)
        self._roundtrip(MODEL_HES1)
        self._roundtrip(MODEL_DIMERISATION)
