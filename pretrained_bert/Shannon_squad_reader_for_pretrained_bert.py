import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField
from allennlp.data.instance import Instance
from pytorch_pretrained_bert import BertTokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register("squad_for_pretrained_bert")
class ShannonSquadReaderForPretrainedBert(DatasetReader):
    def __init__(self,
                 pretrained_bert_model_file: str,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = BertTokenizer.from_pretrained(pretrained_bert_model_file)

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
            for entry in dataset:
                for paragraph in entry["paragraphs"]:
                    paragraph_text = paragraph["context"]

                    for qa in paragraph["qas"]:
                        question_text = qa["question"]
                        is_impossible = qa["is_impossible"]
                        if not is_impossible:
                            answer = qa["answers"][0]
                            orig_answer_text = answer["text"]
                            answer_offset = answer["answer_start"]
                        else:
                            orig_answer_text = ''
                            answer_offset = -1
                        instance = self.text_to_instance(
                            question_text=question_text,
                            paragraph_text=paragraph_text,
                            origin_answer_text=orig_answer_text,
                            answer_offset=answer_offset,
                        )
                        if instance is not None:
                            yield instance

    @staticmethod
    def is_whitespace(char):
        if char == " " or char == "\t" or char == "\r" or char == "\n" or ord(char) == 0x202F:
            return True
        return False

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         paragraph_text: str,
                         origin_answer_text: str = '',
                         answer_offset: int = -1,
                         ) -> Instance:
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if self.is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
        if origin_answer_text:
            start_position = char_to_word_offset[answer_offset]
            end_position = char_to_word_offset[answer_offset + len(origin_answer_text) - 1]
        else:
            start_position = -1
            end_position = -1
        instance = Instance({
            "question_text": MetadataField(question_text),
            "paragraph_text": MetadataField(paragraph_text),
            "origin_answer_text": MetadataField(origin_answer_text),
            "start_position": MetadataField(start_position),
            "end_position": MetadataField(end_position),
            "document_tokens": MetadataField(doc_tokens),
        })
        return instance
