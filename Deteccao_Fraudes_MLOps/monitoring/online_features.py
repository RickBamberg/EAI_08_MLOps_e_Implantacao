# monitoring/online_features.py

import numpy as np

def compute_online_features(transaction, state):

    amount = float(transaction["amount"])
    age = transaction.get("age", 0)
    gender_encoded = transaction["gender_encoded"]
    category_encoded = transaction["category_encoded"]

    total_tx = state["total_tx"]
    sum_amount = state["sum_amount"]
    sum_amount_sq = state["sum_amount_sq"]

    mean = sum_amount / total_tx if total_tx > 0 else 0

    if total_tx > 1:
        variance = (sum_amount_sq / total_tx) - (mean ** 2)
        std = np.sqrt(max(variance, 0))
    else:
        std = 0

    last_5 = list(state["last_5_amounts"])
    rolling_mean = np.mean(last_5) if last_5 else amount

    alert_valor = int(amount > mean + 3 * std)
    valor_relativo_cliente = amount / (mean + 1e-6)

    step = transaction["step"]
    qtd_transacoes = state["step_counts"][step]
    alert_freq = int(qtd_transacoes > 3)

    merchant = transaction["merchant"]
    primeira_tx_merchant = int(state["merchants"][merchant] == 0)

    zipcode_ori = transaction.get("zipcodeOri", 0)
    zipcode_mer = transaction.get("zipMerchant", 0)
    mesma_localizacao = int(zipcode_ori == zipcode_mer)

    num_zipcodes_cliente = len(state["zipcodes"])

    return {
        "amount": amount,
        "age": age,
        "gender_encoded": gender_encoded,
        "category_encoded": category_encoded,
        "qtd_transacoes": qtd_transacoes,
        "alert_freq": alert_freq,
        "alert_valor": alert_valor,
        "valor_relativo_cliente": valor_relativo_cliente,
        "amount_media_5steps": rolling_mean,
        "primeira_tx_merchant": primeira_tx_merchant,
        "mesma_localizacao": mesma_localizacao,
        "num_zipcodes_cliente": num_zipcodes_cliente
    }

