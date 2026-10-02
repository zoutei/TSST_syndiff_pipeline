import json
from types import SimpleNamespace
import pandas as pd
import pytest
from syndiff_pipeline.template_creation.processing.removal_ledger.catalogues import fetch_ps1_cone


class Response:
    def __init__(self,value=None,text=''):
        self.value=value;self.content=text.encode()
    def json(self):return self.value
    def raise_for_status(self):pass


class API:
    def __init__(self,*,repeat=False,short=False):self.repeat=repeat;self.short=short
    def get(self,url,params=None,timeout=None):
        if url.endswith('/metadata.json'):
            return Response([dict(name=n,datatype=t) for n,t in [('objID','long'),('uniquePspsSTid','long'),('raStack','double'),('decStack','double'),('primaryDetection','int')]])
        if '/count' in url:return Response(dict(data=[[3]]))
        page=params['page'];assert params['sort_by']=='uniquePspsSTid.asc'
        header='objID,uniquePspsSTid,raStack,decStack,primaryDetection\n'
        if page==1 or self.repeat:
            return Response(text=header+'9007199254740993,9107199254740993,10,20,0\n9007199254740993,9107199254740994,10.0001,20,1\n')
        return Response(text=header if self.short else header+'9007199254740995,9107199254740995,10.1,20,0\n')


def test_all_pages_lossless_ids_nonprimary_and_split_detection(tmp_path):
    table,meta,path=fetch_ps1_cone(tmp_path,10.,20.,.1,session=API(),page_size=2)
    assert table.objID.tolist()==['9007199254740993','9007199254740993','9007199254740995']
    assert table.uniquePspsSTid.nunique()==3
    assert (table.primaryDetection==0).sum()==2
    assert meta['status']=='complete' and meta['rows']==3
    again,_,_=fetch_ps1_cone(tmp_path,10.,20.,.1,session=SimpleNamespace(),page_size=2)
    pd.testing.assert_frame_equal(table,again)


@pytest.mark.parametrize('bad',['repeat','short'])
def test_truncated_or_repeated_page_cannot_publish(tmp_path,bad):
    with pytest.raises(ValueError):fetch_ps1_cone(tmp_path,10.,20.,.1,session=API(**{bad:True}),page_size=2)
    assert not list(tmp_path.glob('*/manifest.json'))


def test_missing_required_metadata_rejected(tmp_path):
    api=SimpleNamespace(get=lambda *a,**k:Response([dict(name='objID',datatype='long')]))
    with pytest.raises(ValueError,match='schema'):fetch_ps1_cone(tmp_path,10.,20.,.1,session=api)
