import unittest
from unittest.mock import MagicMock
import tempfile
import pandas as pd
from predict.predict.run import TextPredictionModel
from preprocessing.preprocessing import utils


def load_dataset_mock():
    titles = [
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
    ]
    tags = ["php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails",
            "php", "ruby-on-rails"]

    return pd.DataFrame({
        'title': titles,
        'tag_name': tags
    })



class TestTrain(unittest.TestCase):
    # use the function defined above as a mock for utils.LocalTextCategorizationDataset.load_dataset
    utils.LocalTextCategorizationDataset.load_dataset = MagicMock(return_value=load_dataset_mock())

    def test_predict(self):
        
        base = TextPredictionModel.from_artefacts("train/data/artefacts/test/2023-01-05-11-59-17")
        predictions = TextPredictionModel(base.model, base.params, base.labels_to_index).predict(["Is it possible to execute the procedure of a function in the scope of the caller?",
                                        "ruby on rails: how to change BG color of options in select list, ruby-on-rails"],
                                        top_k=1)
                                
        # assert predicted labels
        self.assertEqual(predictions, ["php","ruby-on-rails"])

#"train/data/artefacts/test/2023-01-05-11-59-17/train_output.json"


