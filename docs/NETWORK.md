# Neural Network Architecture of `ts_trade`

This document explains the structure of the temporal convolutional network (TCN) used in the trading engine, provides a visual rundown of the layers (with their neuron counts), and highlights what makes this model unique relative to generic neural-network setups.

---

## 1. High-Level Purpose

The network ingests a sequence of market features—specifically a sliding window of `windowSize × featureCount` values extracted from KuCoin trade bars—and outputs the probability that the next bar’s close will be higher than the current bar’s close. This probability is then fed into a proprietary risk map to produce tradable exposure.

---

## 2. Layer-by-Layer Structure

The model is constructed in `src/model/tcn.ts`. Below is the standard configuration (default values in parentheses):

1. **Input Layer**
   - Shape: `windowSize × featureCount`
   - Default `windowSize = 64`, `featureCount = 12` (each row corresponds to one bar; each column is a feature such as log return, volume, VWAP, etc.).
2. **Residual Blocks (3 by default)**
   - Each block uses:
     - **Conv1D #1**: filters=32/64/64, kernel=3, dilation=1, padding=`same`, activation=`relu`
     - Optional dropout (default rate 0.1) after the first conv
     - **Conv1D #2**: same filter count and kernel size, activation=`relu`
     - **Residual Connection**: 1×1 Conv1D to match input channels (if needed), then added to Conv1D #2 output
     - **Layer Normalization**: stabilizes training by normalizing across features
3. **Intermediate Activation** (after each residual block)
   - ReLU applied to the block output before feeding into the next block.
4. **Global Average Pooling**
   - Collapses the time dimension, producing a single vector of length equal to the number of filters in the last block (e.g., 64).
5. **Dense (“Embedding”) Layer**
   - Units = 64 by default, activation=`relu`
   - Optional dropout applied afterward.
6. **Output Layer**
   - Dense layer with 1 unit, activation=`sigmoid`
   - Produces a probability in `[0, 1]`.

Visually:

```
Input (windowSize × featureCount)
        │
        ├── Residual Block #1 ──► ReLU
        │
        ├── Residual Block #2 ──► ReLU
        │
        ├── Residual Block #3 ──► ReLU
        │
Global Average Pooling (across time)
        │
Dense (64 units, ReLU) ──► Dropout (optional)
        │
Dense (1 unit, Sigmoid) → Probability of Up Move
```

- **Neurons per layer** (default):
  - Conv1D #1: 32 filters × `windowSize` time steps
  - Conv1D #2: 32 filters × `windowSize` time steps
  - Residual Blocks 2 & 3: 64 filters × `windowSize` time steps each
  - Dense Embedding: 64 neurons
  - Output: 1 neuron
  - Total parameter count depends on `featureCount` and exact configuration, but is typically in the low hundreds of thousands.

---

## 3. Why a Temporal Convolutional Network?

1. **Causality**
   - Convolutional filters only look at current and past timesteps (via padding and `same` convolutions), ensuring the network does not “peek” into future data.
2. **Translation invariance**
   - Convolutions ingest patterns independent of the absolute position within the window—useful for detecting recurring price/volume signatures.
3. **Residual Connections**
   - Help avoid vanishing gradients and allow the network to learn identity mappings when deeper transformations aren’t helpful.
4. **Global Average Pooling**
   - Summarises temporal features effectively without an explosion of parameters; behaves similarly to a learned “attention” in a compact form.

---

## 4. How This Network Differs from Typical Setups

1. **Feature-rich inputs**
   - Instead of raw price-only series, every timestep includes engineered features (log returns, spread/close ratios, volume ratios, volatility estimates). This reduces the network’s burden to discover basic relationships from scratch.
2. **Short-window design**
   - Tunable `windowSize` allows the model to be adapted for sparse or dense markets. The ability to shrink the window ensures you can still train with limited history.
3. **Dropout + Normalization in Residual Blocks**
   - Layer normalization inside each residual block stabilizes training when window distributions vary wildly between markets.
4. **Integrated Risk Mapping**
   - The network is optimized to predict a probability, not an immediate trading signal. Downstream scaling (neutral zone, volatility adjustments) is intentionally separated to keep the neural network’s output interpretable.
5. **TFJS-node compatibility adjustments**
   - To keep GPU/CPU compatibility with tfjs-node, dilations > 1 (which can cause gradient issues) were replaced by standard convolutions. This sacrifices some theoretical receptive field but drastically improves stability when training in Node.js runtimes.

---

## 5. How Neural Networks Learn in This Context (Quick Primer)

1. **Forward Pass**
   - The feature window is multiplied by the network’s weights layer by layer to produce a probability.
2. **Loss Calculation**
   - Binary cross-entropy compares the predicted probability with the actual label (whether the next bar closed higher).
3. **Backpropagation**
   - Gradients of the loss with respect to each weight are computed (via automatic differentiation); because of residual connections and normalization, gradient flow remains steady.
4. **Weight Update**
   - Adam optimizer adjusts weights to lower future loss.
5. **Early Stopping**
   - Training halts early if validation loss stops improving, preventing overfitting on limited data.

Over time, the network learns to associate specific sequences of features with upward or downward outcomes, factoring in volume spikes, volatility, and price action.

---

## 6. Customizing the Model

You can tailor the architecture via environment variables (see `README.md`):

| Variable               | Effect                                                  |
|------------------------|---------------------------------------------------------|
| `FEATURE_WINDOW_SIZE`  | Length of the temporal input window                     |
| `TCN_FILTERS`          | Comma-separated filters per block (e.g., `32,64,128`)    |
| `TCN_KERNEL`           | Convolution kernel size                                 |
| `TCN_DROPOUT`          | Dropout rate inside residual blocks and embedding layer |
| `TCN_DENSE_UNITS`      | Size of the dense embedding layer                       |
| `TRAINING_EPOCHS`      | Number of epochs per run                                |

These knobs let you make the network shallower, deeper, wider, or more regularized depending on the symbol’s liquidity and your computational constraints.

---

## 7. Takeaways

- The TCN is structured to balance predictive power with quick retraining—essential for streaming markets.
- Residual blocks with normalization keep training stable across varying market regimes.
- Global averaging and dense layers convert temporal patterns into a single probability output.
- The model’s combination of feature-rich inputs, residual architecture, and dedicated risk mapping downstream makes it well-suited for short-term, probability-driven trading decisions.

Refer to `src/model/tcn.ts` for the exact implementation and `src/model/trainer.ts` for training details.
