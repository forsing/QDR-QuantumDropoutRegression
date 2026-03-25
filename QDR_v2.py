# Quantum Dropout Regression (QDR) for Lottery Prediction
# Lottery prediction generated using a deep variational circuit with Stochastic Dropout regularization.
# Quantum Regression Model with Qiskit

# v2: df.copy(); jači COBYLA + više startova; clip po poziciji (sortirana 7/39); malo duži prozor.

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize

from qiskit_machine_learning.utils import algorithm_globals
import random

# ================= SEED PARAMETERS =================
SEED = 39
random.seed(SEED)
np.random.seed(SEED)
algorithm_globals.random_seed = SEED
# ==================================================


# Use the existing dataframe
df_raw = pd.read_csv('/Users/4c/Desktop/GHQ/data/loto7hh_4586_k24.csv')
# 4586 historical draws of Lotto 7/39 (Serbia)

_MIN_POS = np.array([1, 2, 3, 4, 5, 6, 7], dtype=int)
_MAX_POS = np.array([33, 34, 35, 36, 37, 38, 39], dtype=int)


def quantum_dropout_predict(df):
    df = df.copy()
    cols = ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6', 'Num7']
    predictions = {}

    # Model Hyperparameters
    num_qubits = 1
    num_layers = 4  # Deeper circuit to benefit from dropout
    dropout_rate = 0.2  # 20% chance to drop a gate's influence during training
    train_window = 20

    # 1. Define a Deep Variational Circuit
    x_param = ParameterVector('x', 1)
    # 2 parameters per layer (RZ, RY)
    theta_param = ParameterVector('theta', num_layers * 2)

    qc = QuantumCircuit(num_qubits)
    qc.ry(x_param[0], 0)  # Encoding

    for i in range(num_layers):
        qc.rz(theta_param[i * 2], 0)
        qc.ry(theta_param[i * 2 + 1], 0)

    observable = SparsePauliOp('Z')
    estimator = StatevectorEstimator()

    # Map parameter names to indices for easy manipulation
    # [x, theta_0, theta_1, ..., theta_7]
    all_params = [x_param[0]] + list(theta_param)

    def get_prediction(x_val, weights, dropout_mask=None):
        """
        Computes expectation value. 
        If dropout_mask is provided, specific weights are zeroed out.
        """
        active_weights = weights.copy()
        if dropout_mask is not None:
            active_weights = active_weights * dropout_mask

        param_values = [x_val] + list(active_weights)

        pub = (qc, observable, param_values)
        job = estimator.run([pub])
        ev = job.result()[0].data.evs
        return float(np.real(np.asarray(ev).reshape(-1)[0]))

    for idx, col in enumerate(cols):
        # Feature Engineering: 1 Lag
        df[f'{col}_lag'] = df[col].shift(1)
        df_model = df.dropna().tail(train_window + 1)

        X = df_model[[f'{col}_lag']].values
        y = df_model[col].values

        # Scaling
        scaler_x = MinMaxScaler(feature_range=(0, np.pi))
        scaler_y = MinMaxScaler(feature_range=(-1, 1))

        X_scaled = scaler_x.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        # Training with Stochastic Dropout
        def cost_fn(w):
            mse = 0
            # For each evaluation, we apply a fresh dropout mask (Stochastic)
            # This mimics the behavior of dropout in neural networks
            for i in range(len(X_scaled) - 1):
                # Generate mask: 1 with probability (1-dropout_rate), else 0
                mask = np.random.binomial(1, 1 - dropout_rate, size=len(w))
                # Scale remaining weights by 1/(1-p) to maintain expectation (Inverted Dropout)
                mask = mask / (1 - dropout_rate)

                pred = get_prediction(X_scaled[i][0], w, dropout_mask=mask)
                mse += (pred - y_scaled[i]) ** 2
            return mse / (len(X_scaled) - 1)

        # v2: više iteracija + najbolji od nekoliko startova
        n_w = num_layers * 2
        best_x = None
        best_cost = float("inf")
        for _ in range(5):
            init_w = np.random.uniform(0, 2 * np.pi, n_w)
            res = minimize(
                cost_fn,
                init_w,
                method='COBYLA',
                options={'maxiter': 200, 'rhobeg': 0.25},
            )
            c = float(res.fun)
            if c < best_cost:
                best_cost = c
                best_x = res.x

        # Inference (No dropout, use full weights)
        x_next = X_scaled[-1][0]
        final_y_scaled = get_prediction(x_next, best_x)

        # Inverse scale
        pred_final = scaler_y.inverse_transform(np.array([[final_y_scaled]]))
        lo, hi = int(_MIN_POS[idx]), int(_MAX_POS[idx])
        predictions[col] = int(round(np.clip(pred_final[0][0], lo, hi)))

    return predictions

print("Computing predictions using Quantum Dropout Regression (QDR)...")
q_dr_results = quantum_dropout_predict(df_raw)

# Format for display
q_dr_df = pd.DataFrame([q_dr_results])
# q_dr_df.index = ['Quantum Dropout Regression (QDR) Prediction']

print()
print("Lottery prediction generated using a deep variational circuit with Stochastic Dropout regularization.")
print()
print("Quantum Dropout Regression (QDR) Results:")
print(q_dr_df.to_string(index=True))
print()

"""
Computing predictions using Quantum Dropout Regression (QDR)...

Lottery prediction generated using a deep variational circuit with Stochastic Dropout regularization.

Quantum Dropout Regression (QDR) Results:
   Num1  Num2  Num3  Num4  Num5  Num6  Num7
0     4    x    25    y    30    26    z

(v2: COBYLA 200 iter × 5 startova; train_window 20; očekivana vrednost kao float.)

v2: df.copy(); train_window 15→20; COBYLA do 200 iteracija, 5 slučajnih startova, uzima se najbolji cost; get_prediction vraća čist float iz evs; posle inverzije skaliranja clip po poziciji (1..33 … 7..39).
"""
