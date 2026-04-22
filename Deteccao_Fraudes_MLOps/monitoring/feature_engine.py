def compute_features(transaction, state):

    amount = transaction["amount"]
    total_tx = state["total_tx"]

    mean = (
        state["sum_amount"] / total_tx
        if total_tx > 0 else 0
    )

    media_5 = (
        sum(state["last_5_amounts"]) / len(state["last_5_amounts"])
        if state["last_5_amounts"]
        else amount
    )

    return {
        "age": transaction["age"],
        "amount": amount,
        "valor_relativo_cliente": amount / (mean + 1e-6),
        "amount_media_5steps": media_5,
    }
