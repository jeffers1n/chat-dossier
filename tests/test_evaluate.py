import unittest
from evaluate import score, evaluate

class EvaluationTests(unittest.TestCase):
    def test_strict_category_and_evidence_matching(self):
        expected=[{'category':'decision','evidence_ids':['m1']},{'category':'promise','evidence_ids':['m2']}]
        predicted=[{'category':'decision','evidence_ids':['m1']},{'category':'promise','evidence_ids':['m3']}]
        self.assertEqual(score(expected,predicted),{'expected':2,'predicted':2,'matched':1,'precision':0.5,'recall':0.5,'f1':0.5})
    def test_invalid_quote_is_not_counted(self):
        predictions={'scope':[{'category':'decision','title':'t','evidence':[{'message_id':'m1','quote':'выдуманная цитата'}]}]}
        result=evaluate(predictions)['scope']
        self.assertEqual(result['predicted'],0)
        self.assertEqual(result['rejected_invalid_evidence'],1)

if __name__=='__main__':unittest.main()
