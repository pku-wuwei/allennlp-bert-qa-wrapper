# pylint: disable=no-self-use,invalid-name
import os

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list
from pretrained_bert.Shannon_squad_reader_for_pretrained_bert import ShannonSquadReaderForPretrainedBert


class TestSemanticScholarDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):
        data_path = '/data/nfsdata/home/wuwei/work/allennlp_bert_qa/tests/fixtures/sample_squad_2.0.json'
        reader = SquadReaderForPretrainedBert(
            pretrained_bert_model_file="/data/nfsdata/nlp/BERT_BASE_DIR/uncased_L-12_H-768_A-12")
        instances = ensure_list(reader.read(data_path))

        instance1 = {"question_text": "When did Beyonce start becoming popular?",
                     "paragraph_text": "Beyonc\u00e9 Giselle Knowles-Carter (/bi\u02d0\u02c8j\u0252nse\u026a/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyonc\u00e9's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles \"Crazy in Love\" and \"Baby Boy\".",
                     "origin_answer_text": "in the late 1990s",
                     "start_position": 39,
                     "end_position": 43,
                     "document_tokens": []
                     }

        assert len(instances) == 3
        fields = instances[0].fields
        assert fields["question_text"].metadata == instance1["question_text"]
        assert fields["paragraph_text"].metadata == instance1["paragraph_text"]
        assert fields["origin_answer_text"].metadata == instance1["origin_answer_text"]
        assert fields["start_position"].metadata == instance1["start_position"]
        assert fields['document_tokens'].metadata == instance1["paragraph_text"].split()


if __name__ == '__main__':
    test = TestSemanticScholarDatasetReader()
    test.test_read_from_file()
