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
    "batch_size" : 4
  },
  "trainer": {
    "num_epochs": 20,
    "grad_norm": 5.0,
    "patience": 10,
    "validation_metric": "+em",
    "cuda_device": 0,
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 2
    },
    "optimizer": {
      "type": "adam",
      "betas": [
        0.9,
        0.9
      ]
    }
  }
}
