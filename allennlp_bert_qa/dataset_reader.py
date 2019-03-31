#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author: Wei Wu 
@license: Apache Licence 
@file: dataset_reader.py 
@time: 2019/03/31
@contact: wu.wei@pku.edu.cn
"""
import json
import logging

from overrides import overrides
from typing import Dict

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.fields import TextField, IndexField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, CharacterTokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register("shannon_bert_single_seq")
class ShannonBertSingleSeqReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 content_length_limit: int = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or CharacterTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.content_length_limit = content_length_limit

    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            for line in dataset_file:
                line = line.strip("\n")
                if not line:
                    continue

                example = json.loads(line)
                instance = self.text_to_instance(example['question'], example['passage'],
                                                 example['start_index'], example['end_index'], example['answer'])
                yield instance

    @overrides
    def text_to_instance(self,
                         question: str,
                         passage: str,
                         span_start: int = None,
                         span_end: int = None,
                         answer: str = None
                         ) -> Instance:

        tokenized_question = self._tokenizer.tokenize(question)
        tokenized_passage = self._tokenizer.tokenize(passage)
        util.make_reading_comprehension_instance(tokenized_question, tokenized_passage, self._token_indexers, passage)
        if self.content_length_limit:
            # Take the [CLS] and [SEP] into account
            tokenized_passage = tokenized_passage[: self.content_length_limit - 3]
        tokenized_content = [Token("[CLS]")] + tokenized_question + [Token("[SEP]")] + tokenized_passage + [
            Token("[SEP]")]
        content_field = TextField(tokenized_content, self._token_indexers)
        fields = {'content': content_field}
        metadata = {'question': question, 'passage': passage,
                    'question_tokens': [token.text for token in tokenized_question],
                    'passage_tokens': [token.text for token in tokenized_passage], }
        if span_start and span_end:
            fields['span_start'] = IndexField(span_start, content_field)
            fields['span_end'] = IndexField(span_end, content_field)
            metadata['answer'] = answer
        return Instance(fields)
