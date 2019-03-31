{
  "dataset_reader": {
    "type": "shannon_bert_single_seq",
    "tokenizer": {
      "type": "character"
    },
    "token_indexers": {
      "bert": {
        "type": "bert-pretrained",
        "pretrained_model": "/data/nfsdata/nlp/BERT_BASE_DIR/chinese_L-12_H-768_A-12/",
      }
    }
  },
  "train_data_path": "/home/fangzhiqiang/workspace/gitlab/allennlp/allennlp/tests/fixtures/shannon_sequence_classifier/bert_single_seq_clf/tiny_dataset.jsonl",
  "validation_data_path": "/home/fangzhiqiang/workspace/gitlab/allennlp/allennlp/tests/fixtures/shannon_sequence_classifier/bert_single_seq_clf/tiny_dataset.jsonl",
  "model": {
    "type": "shannon_bert_single_seq_clf",
    "text_field_embedder": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "bert": ["bert", "bert-offsets"]
            },
            "token_embedders": {
                "bert": {
                    "type": "bert-pretrained",
                    "pretrained_model": "/data/nfsdata/nlp/BERT_BASE_DIR/chinese_L-12_H-768_A-12/",
                    "requires_grad": true,
                    "top_layer_only": true
                }
            }
        },
    "classifier_feedforward": {
      "input_dim": 768,
      "num_layers": 1,
      "hidden_dims": [2],
      "activations": ["linear"]
    }
  },
  "iterator": {
        "type": "bucket",
        "sorting_keys": [
            [
                "content",
                "num_tokens"
            ]
        ],
        "batch_size": 32,
        "max_instances_in_memory": 600
    },
    "trainer": {
        "num_epochs": 2,
        "grad_norm": 5,
        "patience": 5,
        "validation_metric": "+accuracy",
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 3e-5
        }
    }
}