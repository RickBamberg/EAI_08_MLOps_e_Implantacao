import numpy as np
import pandas as pd

def calculate_psi(expected, actual, bins=10):

    expected = np.array(expected)
    actual = np.array(actual)

    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    # evitar divisão por zero
    expected_perc = np.where(expected_perc == 0, 0.0001, expected_perc)
    actual_perc = np.where(actual_perc == 0, 0.0001, actual_perc)

    psi = np.sum((actual_perc - expected_perc) *
                 np.log(actual_perc / expected_perc))

    return psi

def calculate_psi_for_dataframe(df_expected, df_actual):

    numeric_cols = df_expected.select_dtypes(include=np.number).columns

    psi_dict = {}

    for col in numeric_cols:

        psi_value = calculate_psi(
            df_expected[col],
            df_actual[col]
        )

        psi_dict[col] = psi_value

    psi_df = pd.DataFrame.from_dict(
        psi_dict,
        orient='index',
        columns=['psi']
    ).sort_values(by='psi', ascending=False)

    return psi_df
