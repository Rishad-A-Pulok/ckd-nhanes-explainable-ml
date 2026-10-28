"""Reproducible CKD label prediction experiments. Run from repository root."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
import joblib

SEED=42
BASE=['age','gender','ethnicity','education_level','poverty_income_ratio','bmi','bp_systolic','bp_diastolic','blood_urea_nitrogen','albumin_serum','phosphorus','bicarbonate','calcium','uric_acid','ever_smoked','current_smoker','diabetes_diagnosed','insulin_use','diabetes_pills']
DIAGNOSTIC=['egfr','albumin_creatinine_ratio','serum_creatinine','urine_albumin','urine_creatinine']
CAT=['gender','ethnicity','education_level','ever_smoked','current_smoker','diabetes_diagnosed','insulin_use','diabetes_pills']
INVALID=['diabetes_pills','ever_smoked','diabetes_diagnosed','education_level']

def prepare(raw, complete_case=True):
    needed=BASE+['ckd_present','ckd_stage']
    absent=[x for x in needed if x not in raw]
    if absent: raise ValueError(f'Missing columns: {absent}')
    d=raw.loc[raw.ckd_stage.notna() & raw.ckd_stage.ne('Unknown')].copy()
    for c in INVALID: d[c]=d[c].replace({7:np.nan,9:np.nan})
    # Structural skip recoding must be checked against the NHANES questionnaire codebook.
    for c in ['insulin_use','diabetes_pills','current_smoker']: d[c]=d[c].fillna(0)
    if complete_case: d=d.dropna(subset=BASE)
    d=d.loc[d.ckd_present.isin([0,1])].reset_index(drop=True)
    return d

def processor(cols):
    categorical=[c for c in cols if c in CAT]
    numeric=[c for c in cols if c not in CAT]
    return ColumnTransformer([
        ('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),
        ('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),categorical)
    ],sparse_threshold=0)

def candidates():
    return {
        'dummy':DummyClassifier(strategy='prior'),
        'logistic_regression':LogisticRegression(max_iter=3000,class_weight='balanced',random_state=SEED),
        'random_forest':RandomForestClassifier(n_estimators=250,min_samples_leaf=2,class_weight='balanced_subsample',random_state=SEED,n_jobs=-1),
        'hist_gradient_boosting':HistGradientBoostingClassifier(max_iter=160,learning_rate=.05,random_state=SEED)
    }

def run_experiment(df, cols, target, out, name):
    out.mkdir(parents=True,exist_ok=True)
    x=df[cols]; y=df[target]
    counts=y.value_counts()
    if len(counts)<2: raise ValueError(f'{name}: fewer than 2 outcome classes')
    if counts.min()<2: raise ValueError(f'{name}: some classes have fewer than 2 observations; combine clinically defensible stages or omit staging')
    strat=y if counts.min()>=2 else None
    xtr,xte,ytr,yte=train_test_split(x,y,test_size=.2,random_state=SEED,stratify=strat)
    minclass=ytr.value_counts().min()
    if minclass<2: raise ValueError(f'{name}: too few training samples in one class for stratified CV')
    folds=min(5,int(minclass))
    cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=SEED)
    scoring='roc_auc' if len(counts)==2 else 'f1_macro'
    rows=[]
    for model_name,model in candidates().items():
        pipe=Pipeline([('preprocess',processor(cols)),('model',model)])
        score=cross_validate(pipe,xtr,ytr,cv=cv,scoring=scoring,n_jobs=1,error_score='raise')
        rows.append({'model':model_name,'cv_mean':float(np.mean(score['test_score'])),'cv_std':float(np.std(score['test_score'])),'folds':folds})
    table=pd.DataFrame(rows).sort_values('cv_mean',ascending=False)
    table.to_csv(out/f'{name}_cv.csv',index=False)
    best_name=table.iloc[0]['model']
    best=Pipeline([('preprocess',processor(cols)),('model',candidates()[best_name])])
    best.fit(xtr,ytr)
    pred=best.predict(xte)
    metrics={'experiment':name,'selected_model':best_name,'selection_metric':scoring,'train_n':len(xtr),'test_n':len(xte),'accuracy':accuracy_score(yte,pred),'balanced_accuracy':balanced_accuracy_score(yte,pred),'macro_f1':f1_score(yte,pred,average='macro',zero_division=0),'weighted_f1':f1_score(yte,pred,average='weighted',zero_division=0)}
    if len(counts)==2:
        labels=sorted(y.unique().tolist()); pos=labels[-1]
        prob=best.predict_proba(xte)[:,list(best.classes_).index(pos)]
        metrics.update({'positive_class':str(pos),'precision':precision_score(yte,pred,pos_label=pos,zero_division=0),'recall':recall_score(yte,pred,pos_label=pos,zero_division=0),'roc_auc':roc_auc_score(yte,prob),'average_precision':average_precision_score(yte,prob,pos_label=pos)})
    (out/f'{name}_metrics.json').write_text(json.dumps(metrics,indent=2,default=str))
    (out/f'{name}_classification_report.txt').write_text(classification_report(yte,pred,zero_division=0))
    pd.DataFrame(confusion_matrix(yte,pred),index=best.classes_,columns=best.classes_).to_csv(out/f'{name}_confusion_matrix.csv')
    # Only run permutation importance on held-out data for interpretation, not feature/model selection.
    imp=permutation_importance(best,xte,yte,scoring=scoring,n_repeats=5,random_state=SEED,n_jobs=1)
    pd.DataFrame({'feature':cols,'importance_mean':imp.importances_mean,'importance_std':imp.importances_std}).sort_values('importance_mean',ascending=False).to_csv(out/f'{name}_permutation_importance.csv',index=False)
    joblib.dump(best,out/f'{name}_model.joblib')
    return metrics

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',default='data/raw_dataset.csv')
    p.add_argument('--out',default='results')
    p.add_argument('--include-staging',action='store_true')
    p.add_argument('--matched-diagnostic',action='store_true')
    args=p.parse_args()
    raw=pd.read_csv(args.raw)
    df=prepare(raw)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    print('Raw records:',len(raw),'Analysis records:',len(df))
    print('Binary counts:',df.ckd_present.value_counts().to_dict())
    results=[run_experiment(df,BASE,'ckd_present',out,'binary_predictive')]
    if args.include_staging:
        print('Stage counts:',df.ckd_stage.value_counts().to_dict())
        results.append(run_experiment(df,BASE,'ckd_stage',out,'stage_predictive'))
    if args.matched_diagnostic:
        absent=[c for c in DIAGNOSTIC if c not in df]
        if absent: raise ValueError('Missing diagnostic columns: '+str(absent))
        matched=df.dropna(subset=DIAGNOSTIC).copy()
        print('Matched diagnostic cohort:',len(matched))
        # Separate model-selection experiments on an identical matched cohort.
        # Same deterministic split across feature sets for binary target.
        results.append(run_experiment(matched,BASE,'ckd_present',out,'matched_predictive'))
        results.append(run_experiment(matched,BASE+DIAGNOSTIC,'ckd_present',out,'matched_diagnostic_label_reconstruction'))
    pd.DataFrame(results).to_csv(out/'summary.csv',index=False)
    print(pd.DataFrame(results).to_string(index=False))

if __name__=='__main__': main()
