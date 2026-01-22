from features.v1.customer_features import build_customer_features
from features.v1.step_features import build_step_features
from features.v1.merchant_features import build_merchant_features
from features.v1.local_features import build_local_features

def build_features(df):
    df = df.copy()

    df = build_customer_features(df)
    df = build_step_features(df)
    df = build_merchant_features(df)
    df = build_local_features(df)

    return df