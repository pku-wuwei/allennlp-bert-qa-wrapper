import importlib

from allennlp.common.testing import ModelTestCase


class BertForQuestionAnsweringTest(ModelTestCase):
    def setUp(self):
        super(BertForQuestionAnsweringTest, self).setUp()
        importlib.import_module('pretrained_bert.dataset_reader')
        self.set_up_model('/data/nfsdata/home/wuwei/work/allennlp_bert_qa/tests/fixtures/sample_train.jsonnet',
                          '/data/nfsdata/home/wuwei/work/allennlp_bert_qa/tests/fixtures/sample_squad_2.0.json')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file, cuda_device=0, gradients_to_ignore={
            'bert_qa_model.bert.pooler.dense.weight', 'bert_qa_model.bert.pooler.dense.bias'})

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()
