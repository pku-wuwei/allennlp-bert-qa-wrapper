{
  "dataset_reader": {
    "type": "squad_for_pretrained_bert",
    "pretrained_bert_model_file": "bert-large-uncased"
  },
  "train_data_path": "https://s3-us-west-2.amazonaws.com/pradeepd-bert-qa-models/sample_data/sample-v2.0.json",
  "validation_data_path": "https://s3-us-west-2.amazonaws.com/pradeepd-bert-qa-models/sample_data/sample-v2.0.json",
  "model": {
    "type": "bert_for_qa",
    "bert_model_type": "bert_large",
    "pretrained_archive_path": "https://s3-us-west-2.amazonaws.com/pradeepd-bert-qa-models/bert-large/bert_large_archive.tar.gz",
    "null_score_difference_threshold": 0.0 
  },
  "iterator": {
    "type": "basic",
    "batch_size": 40
  },

  "trainer": {
    "num_epochs": 1,
    "cuda_device": -1,
    "optimizer": {
      "type": "adam",
      "betas": [0.9, 0.9]
    }
  }
}
