import * as tf from "@tensorflow/tfjs-node";

export interface TcnConfig {
  windowSize: number;
  featureCount: number;
  filtersPerLayer?: number[];
  kernelSize?: number;
  dropoutRate?: number;
  denseUnits?: number;
  learningRate?: number;
}

const defaultConfig = {
  filtersPerLayer: [32, 64, 64],
  kernelSize: 3,
  dropoutRate: 0.1,
  denseUnits: 64,
  learningRate: 1e-3,
} satisfies Required<Omit<TcnConfig, "windowSize" | "featureCount">>;

const buildResidualBlock = (
  input: tf.SymbolicTensor,
  filters: number,
  kernelSize: number,
  dilationRate: number,
  dropoutRate: number,
): tf.SymbolicTensor => {
  const conv1 = tf.layers
    .conv1d({
      filters,
      kernelSize,
      dilationRate,
      padding: "same",
      activation: "relu",
    })
    .apply(input) as tf.SymbolicTensor;

  const maybeDropped =
    dropoutRate > 0
      ? (tf.layers
          .dropout({ rate: dropoutRate })
          .apply(conv1) as tf.SymbolicTensor)
      : conv1;

  const conv2 = tf.layers
    .conv1d({
      filters,
      kernelSize,
      dilationRate,
      padding: "same",
      activation: "relu",
    })
    .apply(maybeDropped) as tf.SymbolicTensor;

  const residual = tf.layers
    .conv1d({ filters, kernelSize: 1, padding: "same" })
    .apply(input) as tf.SymbolicTensor;

  const added = tf.layers.add().apply([residual, conv2]) as tf.SymbolicTensor;
  return tf.layers
    .layerNormalization({ epsilon: 1e-6 })
    .apply(added) as tf.SymbolicTensor;
};

export const buildTcnModel = (config: TcnConfig): tf.LayersModel => {
  const filtersPerLayer =
    config.filtersPerLayer ?? defaultConfig.filtersPerLayer;
  if (filtersPerLayer.length === 0) {
    throw new Error("TCN config requires at least one convolutional block");
  }

  const kernelSize = config.kernelSize ?? defaultConfig.kernelSize;
  const dropoutRate = config.dropoutRate ?? defaultConfig.dropoutRate;
  const denseUnits = config.denseUnits ?? defaultConfig.denseUnits;
  const learningRate = config.learningRate ?? defaultConfig.learningRate;

  const inputs = tf.input({
    shape: [config.windowSize, config.featureCount],
  });

  let x = inputs;

  filtersPerLayer.forEach((filters) => {
    const dilationRate = 1; // tfjs-node gradients do not currently support dilation > 1
    x = buildResidualBlock(x, filters, kernelSize, dilationRate, dropoutRate);
  });

  const pooled = tf.layers
    .globalAveragePooling1d()
    .apply(x) as tf.SymbolicTensor;
  let dense = tf.layers
    .dense({ units: denseUnits, activation: "relu" })
    .apply(pooled) as tf.SymbolicTensor;
  if (dropoutRate > 0) {
    dense = tf.layers
      .dropout({ rate: dropoutRate })
      .apply(dense) as tf.SymbolicTensor;
  }

  const outputs = tf.layers
    .dense({ units: 1, activation: "sigmoid" })
    .apply(dense) as tf.SymbolicTensor;

  const model = tf.model({ inputs, outputs });
  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
};
