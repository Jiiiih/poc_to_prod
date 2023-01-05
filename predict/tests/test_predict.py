import unittest
from unittest.mock import MagicMock
import tempfile
import pandas as pd
from predict.predict.run import TextPredictionModel
from preprocessing.preprocessing import utils


class TestTrain(unittest.TestCase):

    def test_predict(self):

        artefacts_path = "train/data/artefacts/test/2023-01-05-17-54-07"
        
        base = TextPredictionModel.from_artefacts(artefacts_path)
        predictions = base.predict(["Is it possible to execute the procedure of a function in the scope of the caller?",
                                        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
                                        "Is it possible to execute the procedure of a function in the scope of the caller?",
                                        "Is it possible to execute the procedure of a function in the scope of the caller?"],

                                        top_k=4)
                                
        # assert predicted labels
        self.assertEqual(predictions, ["php","ruby-on-rails","php","php"])


