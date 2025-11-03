import * as tf from "@tensorflow/tfjs-node";
import type { FeatureWindow } from "../types.js";

export interface TrainingTensors {
  xs: tf.Tensor3D;
  ys: tf.Tensor2D;
}

export interface TrainingOptions {
  epochs?: number;
  batchSize?: number;
  validationSplit?: number;
  shuffle?: boolean;
}

export const tensorsFromWindows = (
  samples: FeatureWindow[],
): TrainingTensors => {
  if (samples.length === 0) {
    throw new Error("No feature windows available to build tensors");
  }

  const firstSample = samples[0];
  if (!firstSample) {
    throw new Error("No feature windows available to build tensors");
  }

  const windowSize = firstSample.window.length;
  const featureCount = firstSample.window[0]?.length ?? 0;
  if (featureCount === 0) {
    throw new Error("Feature vectors have zero length");
  }

  const flattened = samples.flatMap((sample) => sample.window.flat());
  const tensor = tf.tensor3d(flattened, [
    samples.length,
    windowSize,
    featureCount,
  ]);
  const labels = tf.tensor2d(samples.map((sample) => [sample.label]));

  return { xs: tensor, ys: labels };
};

export const trainModel = async (
  model: tf.LayersModel,
  data: TrainingTensors,
  options: TrainingOptions = {},
) => {
  const { epochs = 10, batchSize = 32, validationSplit = 0.1, shuffle = true } =
    options;

  return model.fit(data.xs, data.ys, {
    epochs,
    batchSize,
    validationSplit,
    shuffle,
    verbose: 0,
    callbacks: [
      tf.callbacks.earlyStopping({
        patience: 3,
        monitor: "val_loss",
      }),
    ],
  });
};
