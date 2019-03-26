{
  "dataset_reader": {
    "type": "squad_for_pretrained_bert",
    "pretrained_bert_model_file": "/data/nfsdata/nlp/BERT_BASE_DIR/uncased_L-12_H-768_A-12"
  },
  "train_data_path": "/data/nfsdata/nlp/datasets/reading_comprehension/squad/dev-v2.0.json",
  "validation_data_path": "/data/nfsdata/nlp/datasets/reading_comprehension/squad/dev-v2.0.json",
  "model": {
    "type": "bert_for_qa",
    "pretrained_archive_path": "/data/nfsdata/nlp/BERT_BASE_DIR/uncased_L-12_H-768_A-12",
    "null_score_difference_threshold": 0.0
  },
  "iterator": {
    "type": "basic",
    "batch_size" : 2
  },
  "trainer": {
    "num_epochs": 20,
    "grad_norm": 5.0,
    "patience": 10,
    "validation_metric": "+em",
    "cuda_device": 0,
    "optimizer": {
      "type": "bert_adam",
      "lr": 1e-5,
    }
  }
}
