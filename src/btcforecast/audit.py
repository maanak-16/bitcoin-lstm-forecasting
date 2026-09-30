"""Audit of the 2023 notebook's evaluation, using the numbers it printed.

The 2023 notebook scaled prices with MinMaxScaler, then converted predictions
back with ``y * (1 / 5.18164146e-05)``. That is wrong twice:

1. The correct inverse of MinMax scaling is ``x / scale_ + data_min_``; the
   ``+ data_min_`` term was dropped, shifting every price down by the
   training-period minimum.
2. The factor used (1 / 5.18e-05) is not the fitted scaler's own factor for
   the target column (scaler.scale_[0] = 4.05414372e-05), so every value was
   also shrunk by about 22%.

Because MAE is a difference of two series converted the same way, the dropped
offset cancels and only the factor matters. The true MAE is therefore the
printed MAE times (5.18164146e-05 / 4.05414372e-05).
"""

HARDCODED_SCALE = 5.18164146e-05       # notebook cell 29
FITTED_SCALE = 4.05414372e-05          # scaler.scale_[0], notebook cell 28 (Open price)
PRINTED_MAE = 1765.903375582367        # notebook cell 317
PRINTED_ACCURACY = 0.7295526495326612  # 1 - MAE / mean(Y_test), notebook cell 317

# First and last "real" test prices printed by the notebook (cell 31), and the
# actual Open prices for those days shown in the notebook's own data table.
PRINTED_FIRST, TRUE_FIRST = 659.73117592, 16625.509766  # 2023-01-02
PRINTED_LAST, TRUE_LAST = 9190.64453806, 27528.955078   # 2023-04-25


def correct(printed_value, data_min):
    """Undo the notebook's conversion and apply the correct inverse transform."""
    scaled = printed_value * HARDCODED_SCALE
    return scaled / FITTED_SCALE + data_min


def audit():
    # Recover the scaler's training minimum from the first test day...
    data_min = TRUE_FIRST - PRINTED_FIRST * HARDCODED_SCALE / FITTED_SCALE
    # ...then check the diagnosis independently on the last test day.
    last_reconstructed = correct(PRINTED_LAST, data_min)
    true_mae = PRINTED_MAE * HARDCODED_SCALE / FITTED_SCALE
    return {
        "recovered_training_min_open": data_min,
        "last_day_reconstructed": last_reconstructed,
        "last_day_actual": TRUE_LAST,
        "reconstruction_error_usd": abs(last_reconstructed - TRUE_LAST),
        "printed_mae_usd": PRINTED_MAE,
        "true_mae_usd": true_mae,
        "printed_accuracy": PRINTED_ACCURACY,
    }


if __name__ == "__main__":
    for k, v in audit().items():
        print(f"{k:<30} {v:,.2f}" if isinstance(v, float) else f"{k:<30} {v}")
