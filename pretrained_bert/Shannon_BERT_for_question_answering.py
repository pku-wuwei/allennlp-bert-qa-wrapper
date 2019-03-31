import collections
import logging
import math
import os
from typing import Dict, List

import torch
from overrides import overrides
from torch.nn import CrossEntropyLoss

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import SquadEmAndF1
from pytorch_pretrained_bert import BertConfig
from pytorch_pretrained_bert import BertForQuestionAnswering as HuggingFaceBertQA
from pytorch_pretrained_bert.tokenization import BasicTokenizer, BertTokenizer

logger = logging.getLogger(__name__)


@Model.register('bert_for_qa')
class ShannonBertForQuestionAnswering(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_archive_path: str,
                 null_score_difference_threshold: float,
                 n_best_size: int = 20,
                 max_answer_length: int = 30,
                 max_query_length: int = 64,
                 max_sequence_length: int = 384,
                 document_stride: int = 128,
                 do_lower_case: bool = True
                 ) -> None:
        super().__init__(vocab)
        config = BertConfig(os.path.join(pretrained_archive_path, 'bert_config.json'))
        vocab_file = os.path.join(pretrained_archive_path, 'vocab.txt')
        self.tokenizer = BertTokenizer(vocab_file)
        self.bert_qa_model = HuggingFaceBertQA(config)
        self._pretrained_archive_path = pretrained_archive_path
        self._null_score_difference_threshold = null_score_difference_threshold  # 如果预测为null的得分减最好的非null得分大于这个分数，就输出null.
        self._n_best_size = n_best_size
        self._max_answer_length = max_answer_length
        self._max_query_length = max_query_length
        self._max_sequence_length = max_sequence_length
        self._document_stride = document_stride
        self._do_lower_case = do_lower_case
        self._verbose_logging = False
        self._squad_metrics = SquadEmAndF1()

    def convert_examples_to_features(self, examples):
        """Loads a data file into a list of `InputBatch`s."""

        feature_index = 0
        features = []
        for (example_index, (question_text, paragraph_text, origin_answer_text, start_position, end_position,
                             document_tokens)) in enumerate(zip(*examples)):
            query_tokens = self.tokenizer.tokenize(question_text)

            if len(query_tokens) > self._max_query_length:
                query_tokens = query_tokens[:self._max_query_length]

            tok_to_orig_index = []  # orig是按空格分开的的正常的词，tok是用WordPieceTokenizer切割过的token
            orig_to_tok_index = []
            all_doc_tokens = []  # document中所有WordPieceTokenizer切割过的token
            for (i, token) in enumerate(document_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = self.tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            # 找WordPiece之后的start_index, end_index
            if len(origin_answer_text) == 0:  # 没答案就指向[CLS]
                tok_start_position = -1
                tok_end_position = -1
            else:  # start指向答案第一个单词的第一个WordPiece，end指向最后一个单词的最后一个WordPiece
                tok_start_position = orig_to_tok_index[start_position]
                if end_position < len(document_tokens) - 1:
                    tok_end_position = orig_to_tok_index[end_position + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position, tok_end_position) = self._improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, origin_answer_text)

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = self._max_sequence_length - len(query_tokens) - 3

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            # 因为有总长度384的限制，passage过长会切成几块
            _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, self._document_stride)

            logger.info(F"len passage: {len(paragraph_text.split())} doc_spans: {doc_spans}")
            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []  # 送BERT的所有token
                token_to_orig_map = {}  # 送BERT的token_index到原来单词index的映射
                token_is_max_context = {}
                segment_ids = []  # 用来区分AB两句的id，A句为0，B句为1

                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                    is_max_context = self._check_is_max_context(doc_spans, doc_span_index, split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)  # 用来区分真实token跟padding，真实为1，padding为0

                # Zero-pad up to the sequence length.
                while len(input_ids) < self._max_sequence_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                assert len(input_ids) == self._max_sequence_length
                assert len(input_mask) == self._max_sequence_length
                assert len(segment_ids) == self._max_sequence_length

                start_position = None  # 确定答案的起始位置跟终止位置在tokens的位置
                end_position = None
                if self.training and len(origin_answer_text) > 0:
                    # For training, if our document chunk does not contain an annotation
                    # we throw it out, since there is nothing to predict.
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    out_of_span = False
                    if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset
                if self.training and len(origin_answer_text) == 0:
                    start_position = 0
                    end_position = 0

                feature = {
                    'feature_index': feature_index,  # 整体doc_span的index
                    'example_index': example_index,  # 问答例子的index
                    'tokens': tokens,  # 送给BERT的tokens
                    'token_to_orig_map': token_to_orig_map,  # 送BERT的token_index到原来单词index的映射
                    'token_is_max_context': token_is_max_context,  # 每个doc_token在这个doc_span里是不是最大的context
                    'input_ids': input_ids,  # tokens对应的indexes
                    'input_mask': input_mask,  # 用来区分真实token跟padding，真实为1，padding为0
                    'segment_ids': segment_ids,  # 用来区分AB两句的id，A句为0，B句为1
                    'start_position': start_position,  # 答案的起始在tokens里面的index
                    'end_position': end_position,  # 答案的终止在tokens里面的index
                }
                features.append(feature)
                feature_index += 1
                if feature_index < 20:
                    logger.info(feature)
        return features

    def get_predictions(self, doc_tokens, all_features, start_logits, end_logits):
        """
        Write final predictions to the json file and log-odds of null if needed.
        Args:
            doc_tokens: list of list of str，每个example对应一个list of str，代表这个passage以空格分开的tokens
            all_features: list of features，原始特征
            start_logits: Tensor，预测出的start_logits
            end_logits: Tensor，预测出的end_logits

        Returns:
            output_dict:
        """

        example_index_to_features = collections.defaultdict(list)  # 每个example对应哪几个feature
        for feature in all_features:
            example_index_to_features[feature['example_index']].append(feature)

        RawResult = collections.namedtuple("RawResult", ["start_logits", "end_logits"])
        feature_index_to_result = []  # 通过doc_span的unique_id找到预测结果
        for start, end in zip(start_logits, end_logits):
            start = start.detach().cpu().tolist()
            end = end.detach().cpu().tolist()
            feature_index_to_result.append(RawResult(start, end))

        _PrelimPrediction = collections.namedtuple("PrelimPrediction", [
            "feature_index", "start_index", "end_index", "start_logit", "end_logit"])

        all_predictions = []  # 记录每个example的预测结果，n_best预测结果，和null结果的得分
        all_nbest_json = []
        scores_diff_json = []

        for (example_index, doc_token) in enumerate(doc_tokens):
            features = example_index_to_features[example_index]

            prelim_predictions = []
            # null答案得分最低的feature对应的null答案得分，feature index，start和end的logit
            score_null = 1000000  # large and positive
            min_null_feature_index = 0  # the paragraph slice with min null score
            null_start_logit = 0  # the start logit at the slice with min null score
            null_end_logit = 0  # the end logit at the slice with min null score
            for (feature_index, feature) in enumerate(features):
                result = feature_index_to_result[feature_index]
                start_indexes = self._get_best_indexes(result.start_logits,
                                                       self._n_best_size)  # 取start得分最高的前20对应的start index
                end_indexes = self._get_best_indexes(result.end_logits, self._n_best_size)  # 取end得分最高的前20对应的end_index
                # if we could have irrelevant answers, get the min score of irrelevant
                feature_null_score = result.start_logits[0] + result.end_logits[0]  # 更新null答案得分最低
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # We could hypothetically create invalid predictions, e.g., predict
                        # that the start of the span is in the question. We throw out all
                        # invalid predictions.
                        # 答案的index必须在passage的范围内，0 < end - start < max_len，答案不能太靠边
                        if start_index >= len(feature['tokens']):
                            continue
                        if end_index >= len(feature['tokens']):
                            continue
                        if start_index not in feature['token_to_orig_map']:
                            continue
                        if end_index not in feature['token_to_orig_map']:
                            continue
                        if not feature['token_is_max_context'].get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > self._max_answer_length:
                            continue
                        prelim_predictions.append(_PrelimPrediction(feature_index=feature_index,
                                                                    start_index=start_index,
                                                                    end_index=end_index,
                                                                    start_logit=result.start_logits[start_index],
                                                                    end_logit=result.end_logits[end_index]))
            prelim_predictions.append(_PrelimPrediction(feature_index=min_null_feature_index,
                                                        start_index=0,
                                                        end_index=0,
                                                        start_logit=null_start_logit,
                                                        end_logit=null_end_logit))
            prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

            _NbestPrediction = collections.namedtuple("NbestPrediction", ["text", "start_logit", "end_logit"])

            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) >= self._n_best_size:
                    break
                feature = features[pred.feature_index]
                if pred.start_index > 0:  # this is a non-null prediction
                    tok_tokens = feature['tokens'][pred.start_index:(pred.end_index + 1)]
                    orig_doc_start = feature['token_to_orig_map'][pred.start_index]
                    orig_doc_end = feature['token_to_orig_map'][pred.end_index]
                    orig_tokens = doc_token[orig_doc_start:(orig_doc_end + 1)]
                    tok_text = " ".join(tok_tokens)

                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")

                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = " ".join(orig_tokens)

                    final_text = self._get_final_text(tok_text, orig_text, self._do_lower_case, self._verbose_logging)
                    if final_text in seen_predictions:  # 删掉已见过的prediction，否则这个n_best就可能有文本相同但位置不同的答案
                        continue

                    seen_predictions[final_text] = True
                else:
                    final_text = ""
                    seen_predictions[final_text] = True

                nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit))
            # if we didn't include the empty option in the n-best, include it
            if "" not in seen_predictions:
                nbest.append(_NbestPrediction(text="", start_logit=null_start_logit, end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) <= 1:
                nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

            total_scores = []
            best_non_null_entry = None  # 得分最高的非null答案
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)
                if entry.text and not best_non_null_entry:
                    best_non_null_entry = entry

            probs = self._compute_softmax(total_scores)  # 由这n个答案的得分算出的概率分布

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                nbest_json.append(output)

            assert len(nbest_json) >= 1

            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - best_non_null_entry.end_logit
            scores_diff_json.append(score_diff)  # 把这个diff存下来，用于确定threshold的值
            if score_diff > self._null_score_difference_threshold:
                all_predictions.append("")
            else:
                all_predictions.append(best_non_null_entry.text)
            all_nbest_json.append(nbest_json)
        output_dict = {
            "score_diff": scores_diff_json,
            "all_predictions": all_predictions,
            "all_nbest_json": all_nbest_json
        }
        return output_dict

    @overrides
    def forward(self,  # type: ignore
                question_text: List[str],
                paragraph_text: List[str],
                origin_answer_text: List[str],
                start_position: List[int],
                end_position: List[int],
                document_tokens: List[str]) -> Dict[str, torch.Tensor]:

        output_dict = {"document_tokens": document_tokens}
        features = self.convert_examples_to_features((question_text, paragraph_text, origin_answer_text,
                                                      start_position, end_position, document_tokens))
        all_input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long).cuda()
        all_input_mask = torch.tensor([f["input_mask"] for f in features], dtype=torch.long).cuda()
        all_segment_ids = torch.tensor([f["segment_ids"] for f in features], dtype=torch.long).cuda()
        start_logits, end_logits = self.bert_qa_model(all_input_ids, all_segment_ids, all_input_mask)

        all_start_positions = torch.tensor([f["start_position"] if f["start_position"] else 0 for f in features], dtype=torch.long).cuda()
        all_end_positions = torch.tensor([f["end_position"] if f["end_position"] else 0 for f in features], dtype=torch.long).cuda()

        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        all_start_positions.clamp_(0, ignored_index)
        all_start_positions.clamp_(0, ignored_index)

        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, all_start_positions)
        end_loss = loss_fct(end_logits, all_end_positions)
        total_loss = (start_loss + end_loss) / 2
        output_dict['loss'] = total_loss

        predictions = self.get_predictions(document_tokens, features, start_logits, end_logits)
        output_dict['start_logits'] = start_logits
        output_dict['end_logits'] = end_logits
        output_dict.update(predictions)
        logger.info(F"predictions: {predictions['all_predictions']}\n origin_answer_text: {origin_answer_text}")
        for i, answer in enumerate(origin_answer_text):
            self._squad_metrics(predictions['all_predictions'][i], [answer])
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._squad_metrics.get_metric(reset)
        return {'em': exact_match, 'f1': f1_score}

    @staticmethod
    def _get_best_indexes(logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = [index_score_pair[0] for index_score_pair in index_and_score[:n_best_size]]
        return best_indexes

    @staticmethod
    def _get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
        """Project the tokenized prediction back to the original text."""

        # When we created the data, we kept track of the alignment between original
        # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
        # now `orig_text` contains the span of our original text corresponding to the
        # span that we predicted.
        #
        # However, `orig_text` may contain extra characters that we don't want in
        # our prediction.
        #
        # For example, let's say:
        #   pred_text = steve smith
        #   orig_text = Steve Smith's
        #
        # We don't want to return `orig_text` because it contains the extra "'s".
        #
        # We don't want to return `pred_text` because it's already been normalized
        # (the SQuAD eval script also does punctuation stripping/lower casing but
        # our tokenizer does additional normalization like stripping accent
        # characters).
        #
        # What we really want to return is "Steve Smith".
        #
        # Therefore, we have to apply a semi-complicated alignment heruistic between
        # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
        # can fail in certain cases in which case we just return `orig_text`.

        def _strip_spaces(text):
            ns_chars = []
            ns_to_s_map = collections.OrderedDict()
            for (i, char) in enumerate(text):
                if char == " ":
                    continue
                ns_to_s_map[len(ns_chars)] = i
                ns_chars.append(char)
            ns_text = "".join(ns_chars)
            return ns_text, ns_to_s_map

        # We first tokenize `orig_text`, strip whitespace from the result
        # and `pred_text`, and check if they are the same length. If they are
        # NOT the same length, the heuristic has failed. If they are the same
        # length, we assume the characters are one-to-one aligned.
        tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

        tok_text = " ".join(tokenizer.tokenize(orig_text))

        start_position = tok_text.find(pred_text)
        if start_position == -1:
            if verbose_logging:
                logger.info(f"Unable to find text: '{pred_text}' in '{orig_text}'")
            return orig_text
        end_position = start_position + len(pred_text) - 1

        (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
        (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

        if len(orig_ns_text) != len(tok_ns_text):
            if verbose_logging:
                logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                            orig_ns_text, tok_ns_text)
            return orig_text

        # We then project the characters in `pred_text` back to `orig_text` using
        # the character-to-character alignment.
        tok_s_to_ns_map = {}
        for (i, tok_index) in tok_ns_to_s_map.items():
            tok_s_to_ns_map[tok_index] = i

        orig_start_position = None
        if start_position in tok_s_to_ns_map:
            ns_start_position = tok_s_to_ns_map[start_position]
            if ns_start_position in orig_ns_to_s_map:
                orig_start_position = orig_ns_to_s_map[ns_start_position]

        if orig_start_position is None:
            if verbose_logging:
                logger.info("Couldn't map start position")
            return orig_text

        orig_end_position = None
        if end_position in tok_s_to_ns_map:
            ns_end_position = tok_s_to_ns_map[end_position]
            if ns_end_position in orig_ns_to_s_map:
                orig_end_position = orig_ns_to_s_map[ns_end_position]

        if orig_end_position is None:
            if verbose_logging:
                logger.info("Couldn't map end position")
            return orig_text

        output_text = orig_text[orig_start_position:(orig_end_position + 1)]
        return output_text

    @staticmethod
    def _compute_softmax(scores):
        """Compute softmax probability over raw logits."""
        if not scores:
            return []

        max_score = None
        for score in scores:
            if max_score is None or score > max_score:
                max_score = score

        exp_scores = []
        total_sum = 0.0
        for score in scores:
            exp_score = math.exp(score - max_score)
            exp_scores.append(exp_score)
            total_sum += exp_score

        probs = []
        for score in exp_scores:
            probs.append(score / total_sum)
        return probs

    def _improve_answer_span(self, doc_tokens, input_start, input_end, orig_answer_text):
        """Returns tokenized answer spans that better match the annotated answer."""

        # The SQuAD annotations are character based. We first project them to
        # whitespace-tokenized words. But then after WordPiece tokenization, we can
        # often find a "better match". For example:
        #
        #   Question: What year was John Smith born?
        #   Context: The leader was John Smith (1895-1943).
        #   Answer: 1895
        #
        # The original whitespace-tokenized answer will be "(1895-1943).". However
        # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
        # the exact answer, 1895.
        #
        # However, this is not always possible. Consider the following:
        #
        #   Question: What country is the top exporter of electronics?
        #   Context: The Japanese electronics industry is the largest in the world.
        #   Answer: Japan
        #
        # In this case, the annotator chose "Japan" as a character sub-span of
        # the word "Japanese". Since our WordPiece tokenizer does not split
        # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
        # in SQuAD, but does happen.
        tok_answer_text = " ".join(self.tokenizer.tokenize(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return new_start, new_end

        return input_start, input_end

    def _check_is_max_context(self, doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""

        # 每个span的最大长为384，span的滑动步长为128，需要确定passage中的每个词在哪个span有最大的context
        # 表现形式为对每个span都有一个长度为len(doc_tokens)的0，1向量
        # Because of the sliding window approach taken to scoring documents, a single
        # token can appear in multiple documents. E.g.
        #  Doc: the man went to the store and bought a gallon of milk
        #  Span A: the man went to the
        #  Span B: to the store and bought
        #  Span C: and bought a gallon of
        #  ...
        #
        # Now the word 'bought' will have two scores from spans B and C. We only
        # want to consider the score with "maximum context", which we define as
        # the *minimum* of its left and right context (the *sum* of left and
        # right context will always be the same, of course).
        #
        # In the example the maximum context for 'bought' would be span C since
        # it has 1 left context and 3 right context, while span B has 4 left context
        # and 0 right context.
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index
