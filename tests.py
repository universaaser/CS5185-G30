"""Focused checks for metric alignment, cache identity, budget input and rule compatibility."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import shutil
import subprocess
import sys

import sweep

import encode
import features
import quality


class BChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.tmp.name)
        cls.video=cls.root/'synthetic.mp4'
        encode.run([quality.FFMPEG,'-v','error','-f','lavfi','-i','testsrc2=size=1280x720:rate=30',
                    '-frames:v','6','-c:v','libx264','-threads','2','-pix_fmt','yuv420p',str(cls.video)])
        cls.metadata,_,_=encode.probe(quality.FFPROBE,cls.video)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_self_comparison_and_timeline_rejection(self):
        quality_result=quality.evaluate(self.video,self.video)
        self.assertEqual(quality_result['psnr_y_db'],'inf')
        self.assertAlmostEqual(quality_result['ms_ssim'],1.,places=6)
        short=self.root/'short.mp4'
        encode.run([quality.FFMPEG,'-v','error','-i',str(self.video),'-frames:v','3',
                    '-c:v','libx264','-threads','2',str(short)])
        with self.assertRaisesRegex(ValueError,'mismatch'):
            quality.evaluate(self.video,short)

    def test_cli_evaluator_tool_resolution(self):
        source = self.root / 'cli.mp4'
        shutil.copyfile(self.video, source)
        # Leave sufficient byte budget for both CLI rates on this six-frame clip.
        with source.open('ab') as stream:
            stream.write(bytes(4_000_000))
        env = os.environ.copy()
        for key in ('CS5185_FFMPEG', 'CS5185_FFPROBE'):
            env.pop(key, None)
        for mode in ('path', 'explicit'):
            output = self.root / mode
            command = [sys.executable, encode.__file__, str(source), '--output', str(output),
                       '--evaluator', 'quality:evaluate']
            if mode == 'explicit':
                for tool in ('ffmpeg', 'ffprobe'):
                    env['CS5185_' + tool.upper()] = 'missing-tool'
                    command.extend(['--' + tool, shutil.which(tool)])
            run = subprocess.run(command, env=env, capture_output=True, text=True)
            jobs = output / 'jobs.jsonl'
            self.assertEqual(run.returncode, 0, jobs.read_text() if jobs.exists() else run.stderr)
            records = [json.loads(line) for line in jobs.read_text().splitlines()]
            self.assertEqual([r['quality']['frames'] for r in records], [6, 6])

    def test_experiment_revalidates_cached_artifacts(self):
        source = self.root / 'resume.mp4'
        shutil.copyfile(self.video, source)
        with patch.object(sweep, 'RUN', self.root / 'run'), patch.object(
                encode, 'encode_job', wraps=encode.encode_job) as encoder:
            def run():
                return sweep.experiment(source, 100, 'fast', 1, 'development')

            record = run()
            output = Path(record['output'])
            self.assertEqual(run(), record)
            self.assertEqual(encoder.call_count, 1)
            output.unlink()
            run()
            with output.open('ab') as stream:
                stream.write(b'damaged')
            run()
            with source.open('ab') as stream:
                stream.write(b'changed source')
            self.assertNotEqual(run()['identity'], record['identity'])
            path = output.with_suffix('.json')
            record = json.loads(path.read_text())
            record.pop('identity')
            path.write_text(json.dumps(record))
            run()
            with patch.object(sweep, 'args', return_value=sweep.args()) as settings:
                settings.return_value.threads = 1
                run()
            self.assertEqual(encoder.call_count, 6)

    def test_budget_endpoints_and_fallback(self):
        rules={'budget_bpp_range':[.07,.28],
               'tree':{'preset':'fast','aq_mode':2,'reason':'test'}}
        for budget in [.07,.2,.28]:
            self.assertEqual(features.rule_decision(rules,[0,0,0,0,budget])[:2], ('fast',2))
        for budget in [.069999,.280001]:
            self.assertEqual(features.rule_decision(rules,[0,0,0,0,budget])[:2], ('medium',1))

    def test_cache_content_identity_and_budget(self):
        rules=self.root/'rules.json'
        rules.write_text(json.dumps({'version':'test','feature_version':features.FEATURE_VERSION,'feature_names':features.NAMES,
                                    'tree':{'feature':4,'threshold':.002,
                                            'left':{'preset':'fast','aq_mode':2,'reason':'low budget test'},
                                            'right':{'preset':'medium','aq_mode':1,'reason':'test fallback'}}}),encoding='utf-8')
        with patch.dict(os.environ,{'CS5185_FEATURE_CACHE':str(self.root/'cache'),'CS5185_RULES':str(rules)}):
            first,timing=features.analyze(self.video)
            second,hit=features.analyze(self.video)
            self.assertFalse(timing['cache_hit']);self.assertTrue(hit['cache_hit']);self.assertEqual(first,second)
            renamed=self.root/'renamed.mp4';shutil.copyfile(self.video,renamed)
            self.assertTrue(features.analyze(renamed)[1]['cache_hit'])
            with renamed.open('ab') as stream:stream.write(b'cache-key-change')
            self.assertFalse(features.analyze(renamed)[1]['cache_hit'])
            low=features.select(self.video,self.metadata,1000)
            high=features.select(self.video,self.metadata,2500)
            self.assertEqual(low['preset'],'fast');self.assertEqual(low['aq_mode'],2)
            self.assertEqual(high['preset'],'medium');self.assertEqual(high['aq_mode'],1)
            self.assertAlmostEqual(high['features']['budget_bpp']/low['features']['budget_bpp'],2.5)
            rules.write_text(json.dumps({'feature_version':'incompatible','feature_names':features.NAMES}),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'do not match'):
                features.select(self.video,self.metadata,1000)

    def test_select_degrades_to_baseline_on_analysis_failure(self):
        with patch.dict(os.environ,{'CS5185_FEATURE_CACHE':str(self.root/'cache-degrade')}):
            missing=self.root/'missing.mp4'
            decision=features.select(missing,self.metadata,1000)
            self.assertEqual((decision['preset'],decision['aq_mode']),('medium',1))
            self.assertEqual(decision['rule_version'],'baseline-fallback')
            self.assertIn('Baseline fallback',decision['reason'])
            self.assertIsNone(decision['source_sha256'])

    def test_select_still_raises_on_non_positive_budget(self):
        with patch.dict(os.environ,{'CS5185_FEATURE_CACHE':str(self.root/'cache-budget')}):
            with self.assertRaisesRegex(ValueError,'positive budget'):
                features.select(self.video,self.metadata,0)


if __name__=='__main__':unittest.main()
