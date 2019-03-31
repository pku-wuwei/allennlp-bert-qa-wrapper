#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author: Wei Wu 
@license: Apache Licence 
@file: model.py 
@time: 2019/03/31
@contact: wu.wei@pku.edu.cn
"""

import logging
from typing import Dict, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy

logger = logging.getLogger(__name__)
BERT_HIDDEN_DIM = 768


@Model.register("shannon_bert_question_answering_model")
class ShannonBertQuestionAnsweringModel(Model):
    """
    This class implements BERT for question answering using HuggingFace implementation

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFiledEmbedder``
        Used to embed the ``content`` ``TextField`` we get as input to the model.
    initializer : ``InitializerApplicator``, optional(default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    classifier_feedforward : ``FeedForward``
    regularizer : ``RegularizerApplicator``
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size('labels')
        self.classifier_feedforward = classifier_feedforward

        if classifier_feedforward.get_input_dim() != BERT_HIDDEN_DIM:
            raise ConfigurationError(F"The input dimension of the classifier_feedforward, "
                                     F"found {classifier_feedforward.get_input_dim()}, must match the "
                                     F" output dimension of the bert embeder, {BERT_HIDDEN_DIM}")
        self.metrics = {
            "accuracy": CategoricalAccuracy()
        }
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    @overrides
    def forward(self,
                content: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        content : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``.
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch

        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes))`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_content = self._text_field_embedder(content)
        # get the [CLS] vector for classification
        bert_cls_vec = embedded_content[:, 0, :]
        logits = self.classifier_feedforward(bert_cls_vec)
        output_dict = {'logits': logits}
        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict['loss'] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``label`` key to the dictionary with the result
        """
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace='labels') for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
