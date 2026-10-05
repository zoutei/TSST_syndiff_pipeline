import json
from pathlib import Path
import pytest
from syndiff_pipeline.template_creation.processing.removal_ledger.campaign import prepare,task_list,update_status


def setup(out):
    (out/'inventory').mkdir()
    for field in ['C4','F1']:
        (out/'inventory'/f'{field}.json').write_text(json.dumps(dict(cells=[dict(cell='skycell.0001.001',fingerprint='same')],original_inputs={})))


def test_shared_cell_task_is_deduplicated_but_keeps_consumers(tmp_path):
    setup(tmp_path);tasks=task_list(tmp_path,['C4','F1'])
    assert len(tasks)==1 and tasks[0]['fields']==['C4','F1']


def test_prepared_pin_cannot_be_changed_or_submitted_jobs_rewritten(tmp_path):
    setup(tmp_path)
    prepare(tmp_path,['C4'],Path('/tmp/pin'), 'a'*40,with_validation=False)
    with pytest.raises(ValueError,match='repoint'):prepare(tmp_path,['C4'],Path('/tmp/other'),'b'*40,with_validation=False)
    (tmp_path/'campaign/submission.json').write_text('{}')
    with pytest.raises(ValueError,match='immutable'):prepare(tmp_path,['C4'],Path('/tmp/pin'),'a'*40,with_validation=False)


def test_cell_versions_and_validation_states_do_not_overwrite_each_other(tmp_path):
    setup(tmp_path);prepare(tmp_path,['C4'],Path('/tmp/pin'),'a'*40,with_validation=False)
    update_status(tmp_path,dict(cell='same',fingerprint='one',field='C4'),'complete')
    update_status(tmp_path,dict(cell='same',fingerprint='two',field='F1'),'failed',error='fixture')
    update_status(tmp_path,dict(cell='same',state_key='validation:C4:same',field='C4'),'validation_complete')
    state=json.loads((tmp_path/'campaign/summary.json').read_text())
    assert len(state)==3 and state['same:one']['status']=='complete'
