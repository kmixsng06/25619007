# 25619007
갑상선 암

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

train = pd.read_csv('/content/drive/MyDrive/open/train.csv')
test = pd.read_csv('/content/drive/MyDrive/open/test.csv')

X = train.drop(columns=['ID', 'Cancer'])
y = train['Cancer']

x_test = test.drop('ID', axis=1)

# OneHotEncoder로 범주형 변수 인코딩
categorical_features = [col for col in X.columns if X[col].dtype == 'object']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = encoder.fit_transform(X[categorical_features])
X_encoded = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_features))

# 수치형 컬럼은 그대로
X_numeric = X.drop(columns=categorical_features).reset_index(drop=True)
X_final = pd.concat([X_numeric, X_encoded], axis=1)

# x_test 인코딩
x_test_encoded = encoder.transform(x_test[categorical_features])
x_test_encoded = pd.DataFrame(x_test_encoded, columns=encoder.get_feature_names_out(categorical_features))

# 수치형 데이터 붙이기
x_test_numeric = x_test.drop(columns=categorical_features).reset_index(drop=True)
x_test_final = pd.concat([x_test_numeric, x_test_encoded], axis=1)

X_train, X_val, y_train, y_val = train_test_split(X_final, y, stratify=y, test_size=0.2, random_state=42)

def train_and_eval(X_tr, y_tr, X_val, y_val, label, model, hyperparams=None):
    if hyperparams:
        model = GridSearchCV(model, hyperparams, cv=3, n_jobs=-1, verbose=1, scoring='f1')
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)
    print(f"[{label}] Validation F1-score: {f1:.4f}")
    return model, f1

# (1) SMOTE 미적용
model_raw = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
f1_raw = train_and_eval(X_train, y_train, X_val, y_val, "RAW", model_raw)[1]

# (2) SMOTE 적용
smote = SMOTE(random_state=42)

# SMOTE 적용하여 오버샘플링
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 결측치 처리 (필요한 경우)
X_train_smote = X_train_smote.fillna(X_train_smote.mean())
y_train_smote = y_train_smote.fillna(y_train_smote.mode()[0])

model_smote = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
f1_smote = train_and_eval(X_train_smote, y_train_smote, X_val, y_val, "SMOTE", model_smote)[1]

# SMOTE 적용 여부에 따라 최종 학습 데이터 구성
if f1_smote >= f1_raw:
    smote_full = SMOTE(random_state=42)

    # OneHotEncoder로 범주형 변수 인코딩
    categorical_features = [col for col in X.columns if X[col].dtype == 'object']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    X_encoded = encoder.fit_transform(X[categorical_features])
    X_encoded = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_features))
    
    # 수치형 데이터 합치기
    X_numeric = X.drop(columns=categorical_features).reset_index(drop=True)
    X_final = pd.concat([X_numeric, X_encoded], axis=1)
    
    # SMOTE 적용
    X_final, y_final = smote_full.fit_resample(X_final, y)
else:
    # SMOTE 미적용
    X_final, y_final = X, y

# 최종 모델 학습
final_model = XGBClassifier(random_state=42)
final_model.fit(X_final, y_final)

# 예측 결과
final_pred = final_model.predict(x_test_final)

submission = pd.read_csv('/content/drive/MyDrive/open/sample_submission.csv')
submission['Cancer'] = final_pred

# 결과 저장
submission.to_csv('/content/drive/MyDrive/baseline_submit1.csv', index=False)
