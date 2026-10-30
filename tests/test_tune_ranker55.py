"""Real tiny CPU fits, NPZ alignment and validation-only model selection."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ranker55
import tune_ranker55 as tuning
from label_reference_messages import label_question, load_references
from message_dataset import load_message_dataset
from test_message_retrieval import write_fixture


class TuningTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        write_fixture(self.root)
        base = self.root / 'data/raw/personamem_v2'
        history_path = base / 'data/chat_history_32k/linked_history.json'
        history = json.loads(history_path.read_text())
        history['chat_history'].extend({'role':'user', 'content':f'Additional original message {i}'} for i in range(50))
        history_path.write_text(json.dumps(history), encoding='utf-8')
        (base/'data/chat_history_32k/val_history.json').write_text(json.dumps(history), encoding='utf-8')
        for split in ('train','val','benchmark'):
            path = base/f'benchmark/text/{split}.csv'
            with path.open(encoding='utf-8',newline='') as stream:
                rows = list(csv.DictReader(stream))
            rows[0]['related_conversation_snippet'] = json.dumps([{'role':'user','content':'I prefer tea.'}])
            if split == 'val':
                rows[0].update(persona_id='8',chat_history_32k_link='data/chat_history_32k/val_history.json')
            with path.open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (self.root/'data/splits/v2/val_ids.json').write_text(json.dumps(['v2:val:00000']))
        self.paths, self.arrays = {}, {}
        for split in ('train','val'):
            qs, memories, sources = load_message_dataset('v2',[split],root=self.root)
            refs=load_references(qs,sources,self.root)
            labels=label_question(qs[0],memories,refs[qs[0]['example_id']])
            n=len(qs[0]['memory_ids'])
            X=np.random.default_rng(31).normal(size=(n,55)).astype(np.float32)
            data=dict(X=X,y=np.array([int(m['label']==2) for m in labels['memories']],dtype=np.int8),
                      offsets=np.array([0,n]),feature_names=np.array(ranker55.FEATURE_NAMES),
                      question_ids=np.array([qs[0]['example_id']]),persona_ids=np.array([qs[0]['persona_id']]),
                      message_ids=np.array(qs[0]['memory_ids']),
                      metadata_json=np.array(json.dumps(dict(dataset='v2',split=split,memory_unit='single_message',
                                                            statistics_fingerprint='frozen-training-stats',source_sha256=sources))))
            if split=='val':
                data['y']=np.full(n,-1,dtype=np.int8)  # deliberately not usable as training labels
            path=self.root/f'data/ranker55/v2/{split}/features.npz'
            path.parent.mkdir(parents=True)
            np.savez(path,**data)
            self.paths[split],self.arrays[split]=path,data

    def test_mapped_and_compressed_bundles_equal_original_arrays(self):
        for compressed in (False,True):
            path=self.root/'sample.npz'
            (np.savez_compressed if compressed else np.savez)(path,**self.arrays['train'])
            with tuning.Bundle(path) as bundle:
                for key,value in self.arrays['train'].items():
                    np.testing.assert_array_equal(bundle.read(key),value)
                self.assertEqual(bundle.metadata['split'],'train')

    def test_default_grid_has_27_valid_unique_trials_and_default_first(self):
        grid=tuning.parameter_grid([50,100,200],[(15,4),(31,5),(63,6)],[1.,5.,10.])
        self.assertEqual(len(grid),27)
        self.assertEqual(grid[0],dict(min_child_samples=100,num_leaves=31,max_depth=5,reg_lambda=5.))
        self.assertEqual(ranker55.training_parameters()['device_type'], 'cpu')
        for bad in ({'min_child_samples':0},{'num_leaves':63,'max_depth':5},{'reg_lambda':float('nan')}):
            with self.assertRaises(ValueError):
                ranker55.training_parameters(**bad)

    def test_metrics_ties_missing_references_and_nonforget_macro(self):
        targets=dict(questions=[dict(example_id='q0',persona_id='p',question_type='a'),
                                dict(example_id='q1',persona_id='p',question_type='ask_to_forget')],
                     offsets=np.array([0,12,24]),positive=np.array([0]*11+[1]+[1]+[0]*11,dtype=bool),
                     ideal_counts=np.array([2,1]),complete=np.array([False,True]),missing=np.array([1,0]))
        rows=tuning.score_questions(np.zeros(24),np.zeros(24),targets)
        stats=tuning.aggregate(rows)
        self.assertEqual(stats['nonforget_type_macro']['hit_at_10'],0)
        self.assertEqual(stats['all_questions']['hit_at_10'],.5)
        self.assertEqual(stats['complete_reference_nonforget_type_macro']['questions'],0)
        scores=np.zeros(24)
        scores[11]=1
        rows=tuning.score_questions(scores,np.zeros(24),targets)
        self.assertAlmostEqual(rows[0]['ndcg_at_10'],1/(1+1/np.log2(3)))
        self.assertEqual(tuning.aggregate(rows)['nonforget']['tail_rescue_over_cosine_at_10'],1)

    def run_tiny(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return tuning.run(self.paths['train'],self.paths['val'],self.root/'models/ranker55.txt',
                              self.root/'results/tuning',tuning.parameter_grid([1],[(3,2)],[1.]),root=self.root)

    def test_real_fit_saves_evaluator_model_and_evaluates_without_training_on_val_y(self):
        report=self.run_tiny()
        self.assertEqual(report['status'],'complete')
        self.assertEqual(report['validation_questions'],1)
        self.assertEqual(report['selected_trial'],0)  # improvements do not clear the frozen threshold
        self.assertFalse(report['test_used'])
        self.assertTrue((self.root/'models/ranker55.txt').is_file())
        self.assertTrue((self.root/'results/tuning/trials.csv').is_file())
        import lightgbm as lgb
        model=lgb.Booster(model_str=(self.root/'models/ranker55.txt').read_text())
        self.assertEqual(model.feature_name(),ranker55.FEATURE_NAMES)

    def test_wrong_statistics_and_persona_overlap_are_rejected(self):
        data=self.arrays['val']
        metadata=json.loads(str(data['metadata_json'].item()))
        data['metadata_json']=np.array(json.dumps({**metadata,'statistics_fingerprint':'wrong'}))
        np.savez(self.paths['val'],**data)
        with self.assertRaisesRegex(ValueError,'frozen training corpus'):
            self.run_tiny()
        data['metadata_json']=np.array(json.dumps(metadata))
        data['persona_ids']=self.arrays['train']['persona_ids']
        np.savez(self.paths['val'],**data)
        with self.assertRaisesRegex(ValueError,'personas overlap'):
            self.run_tiny()


if __name__=='__main__':
    unittest.main()
