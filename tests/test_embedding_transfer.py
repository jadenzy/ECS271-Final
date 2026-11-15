import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_embedding_ab_transfer import DATASETS, FrozenHistories, accept_protocol, answer_scores
from raw_datasets import memories_for_question


class TransferTests(unittest.TestCase):
    def test_four_orders_restore_answer_texts_and_locomo_stays_single_pass(self):
        labels=list('abcd')
        conditional=dict(zip(labels,[-1.,-2.,-3.,-4.]))
        self.assertEqual(answer_scores(conditional,labels,1),[-4.,-1.,-2.,-3.])
        np.testing.assert_array_equal(np.sum([answer_scores(conditional,labels,s) for s in range(4)],axis=0),[-10.]*4)
        self.assertEqual([DATASETS[d][1] for d in ['v1','v2','locomo']],[4,4,1])
        ten=list('abcdefghij')
        self.assertEqual(answer_scores(dict(zip(ten,range(10))),ten,0),list(range(10)))
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            answer_scores({'a':float('nan')},['a'],0)

    def test_resume_rejects_changed_protocol_duplicates_and_rotation_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'protocol.json'
            protocol=dict(rotations=4,memory_tokens=2048)
            self.assertEqual(accept_protocol(path,protocol,[],['q']),set())
            row=dict(example_id='q',shifts=[{}]*4)
            self.assertEqual(accept_protocol(path,protocol,[row],['q']),{'q'})
            with self.assertRaisesRegex(ValueError,'differs'):
                accept_protocol(path,{**protocol,'memory_tokens':1024},[row],['q'])
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                accept_protocol(path,protocol,[row,row],['q'])
            with self.assertRaisesRegex(ValueError,'count differs'):
                accept_protocol(path,protocol,[{**row,'shifts':[{}]}],['q'])

    def test_packing_respects_v1_negative_cutoff_and_uses_highest_score(self):
        q=dict(example_id='q',dataset='v1',history_id='v1:h',source_end_index=-1,question='Why?',
               options=[dict(label=l,text=l.upper()) for l in 'abcd'])
        contexts={'v1:h':[dict(role='system',content='Profile'),dict(role='user',content='first'),
                          dict(role='assistant',content='answer'),dict(role='user',content='FUTURE')]}
        members=memories_for_question(q,contexts)
        mids=np.array([m['message_id'] for m in members])
        data=SimpleNamespace(qids=['q'],offsets=np.array([0,2]),bundle=SimpleNamespace(read=lambda _:mids))
        histories=FrozenHistories([q],contexts,data,np.array([.1,.9]))
        config=dict(baseline='selector',system_prompt='Instructions',user_template='{question}\n{options}',context_tokens=21)
        class Server:
            actual_context=21
            verified=False
            def render_prompt(self,messages):
                count=messages[1]['content'].count('[Message ') if len(messages)==3 else 0
                return 'prompt',10+10*count
            def verify_selector_prompt(self,*args):
                self.verified=True
        server=Server()
        history,source=histories.get(q,server,config)
        self.assertEqual([m['content'] for m in history],['answer'])
        self.assertEqual(history[0]['source_message_index'],2)
        self.assertEqual(source['packed_memory_tokens'],10)
        self.assertTrue(server.verified)
        wrong=SimpleNamespace(qids=['q'],offsets=np.array([0,3]),
            bundle=SimpleNamespace(read=lambda _:np.concatenate([mids,np.array(['future'])])))
        with self.assertRaisesRegex(ValueError,'visible message pool'):
            FrozenHistories([q],contexts,wrong,np.array([.1,.9,1.]))


if __name__=='__main__':
    unittest.main()
