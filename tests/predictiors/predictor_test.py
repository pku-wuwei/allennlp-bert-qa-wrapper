#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author: Wei Wu 
@license: Apache Licence 
@file: predictor_test.py 
@time: 2019/03/27
@contact: wu.wei@pku.edu.cn
"""
from unittest import TestCase

from pytest import approx
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

# required so that our custom model + predictor + dataset reader
# will be registered by name
import pretrained_bert


class TestPaperClassifierPredictor(TestCase):
    def test_uses_named_inputs(self):
        inputs = {
            "question_text": "Interferring Discourse Relations in Context",
            "paragraph_text": (
                    "We investigate various contextual effects on text "
                    "interpretation, and account for them by providing "
                    "contextual constraints in a logical theory of text "
                    "interpretation. On the basis of the way these constraints "
                    "interact with the other knowledge sources, we draw some "
                    "general conclusions about the role of domain-specific "
                    "information, top-down and bottom-up discourse information "
                    "flow, and the usefulness of formalisation in discourse theory."
            )
        }

        archive = load_archive('tests/fixtures/model.tar.gz')
        predictor = Predictor.from_archive(archive, 'paper-classifier')

        result = predictor.predict_json(inputs)

        label = result.get("all_predictions")
        assert label in {'AI', 'ML', 'ACL'}

