import importlib.util
from pathlib import Path
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]
def controller():
 spec=importlib.util.spec_from_file_location('s24_controller',ROOT/'scripts/colab_job.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_chunk_assembly_preserves_bytes_and_cleans(tmp_path,monkeypatch):
 m=controller();src=tmp_path/'archive.zip';data=b'0123456789abcdefg';src.write_bytes(data);remote=tmp_path/'remote';remote.mkdir()
 def upload(command,**kwargs):Path(command[-1]).write_bytes(Path(command[-2]).read_bytes());return subprocess.CompletedProcess(command,0,'','')
 monkeypatch.setattr(m,'colab_binary',lambda:'fake-colab');monkeypatch.setattr(m,'run_cmd',upload)
 monkeypatch.setattr(m,'remote_exec',lambda session,script,**kwargs:subprocess.run(['bash','-c',script],capture_output=True,text=True,check=True).stdout)
 target=remote/'bundle.zip';m.upload_scene_archive('test',src,tmp_path,chunk_bytes=5,remote_path=str(target))
 assert target.read_bytes()==data and src.read_bytes()==data
 assert not list(remote.glob('*.part-*')) and not list(tmp_path.glob('scene-upload-*'))
 assert (tmp_path/'scene_upload.json').exists()

def test_upload_error_is_redacted_and_no_verified_record(tmp_path,monkeypatch):
 m=controller();src=tmp_path/'archive.zip';src.write_bytes(b'abcdef')
 def fail(*args,**kwargs):raise subprocess.CalledProcessError(1,['upload'],stderr='bad?colab-runtime-proxy-token=SECRET')
 monkeypatch.setattr(m,'colab_binary',lambda:'fake-colab');monkeypatch.setattr(m,'run_cmd',fail)
 with pytest.raises(RuntimeError) as exc:m.upload_scene_archive('test',src,tmp_path,chunk_bytes=3)
 assert 'SECRET' not in str(exc.value)
 assert not (tmp_path/'scene_upload.json').exists() and not list(tmp_path.glob('scene-upload-*'))
