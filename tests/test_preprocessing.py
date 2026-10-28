import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
from train import BASE,prepare

def test_unknown_removed_and_codes_replaced():
    d=pd.DataFrame([{**{c:1 for c in BASE},'ckd_present':1,'ckd_stage':'Stage 1'},
                    {**{c:1 for c in BASE},'ckd_present':0,'ckd_stage':'Unknown'}])
    d.loc[0,'ever_smoked']=7
    assert len(prepare(d))==0
    d.loc[0,'ever_smoked']=1
    assert len(prepare(d))==1
