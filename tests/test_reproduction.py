"""Portable orchestration tests; no scientific training, inference or scoring."""
import argparse
import csv
from datetime import date,timedelta
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import TestCase,main,mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reproduction import common,exports
from reproduction.cli import job_plan


class ReproductionTests(TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='jgr_path_tests_');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.data=self.root/'data';self.out=self.root/'output'
        self.data.mkdir();self.layout=common.Layout(self.data,self.out,'computed')

    def test_outputs_cannot_overlap_inputs(self):
        with self.assertRaises(ValueError):common.Layout(self.data,self.data/'outputs')
        with self.assertRaises(ValueError):common.Layout(self.data,self.root)

    def test_preview_has_no_side_effects(self):
        result=subprocess.run([sys.executable,str(common.SOFTWARE/'run_workflow.py'),'predict',
            '--data-root',str(self.data),'--output-root',str(self.out),'--models','M3'],capture_output=True,text=True,check=True)
        plan=json.loads(result.stdout)
        self.assertFalse(plan['execute']);self.assertFalse(self.out.exists())
        self.assertEqual(plan['source'],'computed')
        self.assertEqual(len(plan['steps'][0]['commands']),2)

    def test_full_blocks_before_any_rf_fit_if_weights_missing(self):
        p=self.data/'datasets/cl10_mapping_v2/manifest.json';p.parent.mkdir(parents=True);p.write_text('{}')
        result=subprocess.run([sys.executable,str(common.SOFTWARE/'run_workflow.py'),'full','--execute',
            '--data-root',str(self.data),'--output-root',str(self.out)],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Missing M1/best.pt',result.stderr)
        self.assertFalse((self.layout.project/'rf_analysis/results_1000').exists())

    def test_computed_workspace_does_not_copy_archived_outputs(self):
        for rel in ['datasets/cl10_mapping_v2/manifest.json','rf_analysis/results_1000/task_summary.csv',
                    'rf_pairs/results/summary.csv','baseline/final_comparison/recall_matched_test_summary.csv']:
            p=self.data/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
        self.layout.prepare()
        self.assertTrue((self.layout.project/'datasets/cl10_mapping_v2/manifest.json').exists())
        self.assertFalse((self.layout.project/'rf_pairs/results/summary.csv').exists())
        self.assertFalse((self.layout.project/'baseline/final_comparison/recall_matched_test_summary.csv').exists())

    def test_shared_inputs_are_restored_without_repopulating_computed_outputs(self):
        p=self.data/'datasets/cl10_mapping_v2/manifest.json';p.parent.mkdir(parents=True);p.write_text('{}')
        original=self.data/'rf_analysis/results_1000/japan_reference.csv';original.parent.mkdir(parents=True);original.write_text('same input\n')
        aliases={
            'rf_pairs/reference/japan.csv':dict(source=str(original.relative_to(self.data)),sha256=common.sha(original)),
            'rf_pairs/results/old.csv':dict(source=str(original.relative_to(self.data)),sha256=common.sha(original))}
        common.dump(self.data/'file_aliases.json',dict(aliases=aliases))
        self.layout.prepare()
        self.assertEqual((self.layout.project/'rf_pairs/reference/japan.csv').read_text(),'same input\n')
        self.assertFalse((self.layout.project/'rf_pairs/results/old.csv').exists())
        self.assertFalse((self.layout.project/'rf_analysis/results_1000/japan_reference.csv').exists())

    def test_workspace_rejects_different_source_mode(self):
        p=self.data/'datasets/cl10_mapping_v2/manifest.json';p.parent.mkdir(parents=True);p.write_text('{}')
        self.layout.prepare()
        other=common.Layout(self.data,self.out,'archived')
        with self.assertRaisesRegex(ValueError,'different configuration'):other.prepare()

    def test_rf_plan_uses_local_reference_not_original_host(self):
        args=argparse.Namespace(workers=4)
        command=job_plan(self.layout,args,'rf-accuracy')[0]
        text=' '.join(command)
        self.assertIn(str(self.layout.project/'reference/04a_rf_full_and_stability.py'),text)
        self.assertNotIn('/share/home',text)
        self.assertNotIn('/1 课题/',text)

    def test_source_runner_supports_multiprocessing_spawn(self):
        script=self.root/'spawn_probe.py'
        script.write_text('''import argparse,json,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
def work(x):return x+1
def main():
 p=argparse.ArgumentParser();p.add_argument('--out');a=p.parse_args()
 with ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn')) as pool:
  result=list(pool.map(work,[1,2]))
 with open(a.out,'w') as f:json.dump(result,f)
if __name__=='__main__':main()
''')
        out=self.root/'probe.json'
        subprocess.run([sys.executable,str(common.SOFTWARE/'reproduction/run_source.py'),
            '--script',str(script),'--','--out',str(out)],check=True,capture_output=True,text=True)
        self.assertEqual(json.loads(out.read_text()),[2,3])

    def test_precision_and_recall_are_read_from_respective_protocols(self):
        rows=[dict(row=i,protocol=p,precision='.8' if p=='target_only' else '.1',
                   recall='.6' if p=='prediction_only' else '.2')
              for i in range(2) for p in ['neither','target_only','prediction_only','both']]
        run=self.root/'run';run.mkdir();(run/'best.pt').write_bytes(b'synthetic checkpoint')
        for ident in ['M5','M6','M7']:
            folder=self.layout.evaluation(ident,'test');folder.mkdir(parents=True)
            common.write(folder/'by_sample.csv',rows)
            common.dump(folder/'evaluation_manifest.json',dict(prediction_checkpoint_sha256=common.sha(run/'best.pt'),
                output_sha256={'by_sample.csv':common.sha(folder/'by_sample.csv')}))
        with mock.patch.dict(common.PROTOCOL,{'expected_splits':{'test':{'sample_count':2}}}), \
             mock.patch.object(self.layout,'check_model',return_value=run):
            output=exports.controls(self.layout)
        self.assertTrue(all(r['precision_mean_pct']==80 and r['recall_mean_pct']==60 for r in output))

    def test_core_requires_every_sensitivity_setting(self):
        folder=self.layout.sensitivity('M1','test');folder.mkdir(parents=True)
        run=self.root/'run';run.mkdir();(run/'best.pt').write_bytes(b'synthetic checkpoint')
        rows=[dict(row=i,pred_threshold=100,d=3) for i in range(2)]
        common.write(folder/'per_window.csv',rows)
        common.dump(folder/'done.json',dict(complete=True,identity={'checkpoint_sha256':common.sha(run/'best.pt')},
                    files_sha256={'per_window.csv':common.sha(folder/'per_window.csv')}))
        with mock.patch.dict(common.PROTOCOL,{'expected_splits':{'test':{'sample_count':2}}}), \
             mock.patch.object(self.layout,'check_model',return_value=run):
            with self.assertRaisesRegex(ValueError,'Missing sensitivity settings'):exports.core_results(self.layout)

    def test_population_and_repeat_sd_are_not_interchanged(self):
        self.assertEqual(exports.moments([1,3]),(2,1))
        self.assertAlmostEqual(exports.moments([1,3],ddof=1)[1],2**.5)

    def test_core_summaries_are_regenerated_from_per_sample_rows(self):
        import numpy as np
        run=self.root/'run';run.mkdir();(run/'best.pt').write_bytes(b'synthetic checkpoint')
        common.dump(self.layout.project/'manuscript/source_summaries/reproduction_grid.json',{'synthetic':True})
        for ident in exports.CORE:
            folder=self.layout.sensitivity(ident,'test');folder.mkdir(parents=True)
            rows=[]
            for setting in common.PROTOCOL['sensitivity_settings']:
                for i in range(2):
                    r=dict(model=common.MODELS[ident]['source_model'],split='test',row=i,
                        target_start='2017-01-0'+str(i+1),target_end='2018-03-2'+str(i+6),
                        pred_threshold=setting['pred_threshold'],d=setting['d'])
                    r.update({field: .2+.2*i for field in exports.STAT_FIELDS});rows.append(r)
            common.write(folder/'per_window.csv',rows)
            np.savez_compressed(folder/'six_maps.npz',predictions=np.zeros((2,2,2),dtype=np.float32),targets=np.ones((2,2,2),dtype=np.float32))
            common.dump(folder/'done.json',dict(complete=True,identity={'checkpoint_sha256':common.sha(run/'best.pt')},
                files_sha256={n:common.sha(folder/n) for n in ['per_window.csv','six_maps.npz']}))
        with mock.patch.dict(common.PROTOCOL,{'expected_splits':{'test':{'sample_count':2}},'nonoverlap_indices':[0,1]}), \
             mock.patch.object(self.layout,'check_model',return_value=run):
            rows,windows,root=exports.core_results(self.layout)
        formal=next(r for r in rows if r['model']=='MAG' and r['cohort']=='daily' and r['pred_threshold']==100 and r['d']==3)
        self.assertAlmostEqual(formal['precision_mean'],.3);self.assertAlmostEqual(formal['precision_sd'],.1)
        self.assertTrue((root/'six_window_maps.npz').exists());self.assertEqual(len(rows),80)

    def test_cloud_import_checks_downloaded_bytes(self):
        cloud=self.root/'cloud';cloud.mkdir()
        src=cloud/'formal_checkpoints_required.json';src.write_text('{}')
        common.dump(cloud/'collection_report.json',dict(complete=False,evaluations=[],copied_files=[
            dict(destination='/cloud/export/formal_checkpoints_required.json',sha256=common.sha(src))]))
        src.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'Downloaded file differs'):common.import_cloud(self.layout,cloud)

    def test_cloud_import_selects_frozen_prediction_identity(self):
        cloud=self.root/'cloud';cloud.mkdir();records=[];evaluations=[]
        cfg=cloud/'formal_checkpoints_required.json';cfg.write_text('{}')
        records.append(dict(destination='/cloud/export/'+cfg.name,sha256=common.sha(cfg)))
        expected=common.MODELS['M1']['analysis_identities']['validation']['prediction_sha256']
        for label,prediction in [('historical','0'*64),('formal',expected)]:
            rel=Path('evaluations/M1/validation/standard')/label;folder=cloud/rel;folder.mkdir(parents=True)
            (folder/'by_sample.csv').write_text(label+'\n')
            common.dump(folder/'prediction_manifest.json',dict(predictions_sha256=prediction))
            for f in folder.iterdir():records.append(dict(destination='/cloud/export/'+str(f.relative_to(cloud)),sha256=common.sha(f)))
            evaluations.append(dict(model='M1',split='validation',kind='standard',destination='/cloud/export/'+str(rel)))
        common.dump(cloud/'collection_report.json',dict(complete=False,copied_files=records,evaluations=evaluations))
        common.import_cloud(self.layout,cloud)
        self.assertEqual((self.layout.evaluation('M1','validation')/'by_sample.csv').read_text(),'formal\n')
        self.assertEqual(len(common.load(self.layout.out/'cloud_import.json')['excluded_nonfinal_deliveries']),1)


if __name__=='__main__':main(verbosity=2)
