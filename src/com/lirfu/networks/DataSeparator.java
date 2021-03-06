package com.lirfu.networks;

import com.lirfu.graphicslib.matrix.IMatrix;

import java.util.ArrayList;

/**
 * Created by lirfu on 26.08.17..<br>
 * Class implements separation of input-output data pairs into the training and test sets of data pairs.
 */
public class DataSeparator {
    public static SeparatedData simpleData(IMatrix[] inputs, IMatrix[] outputs) {
        return new SeparatedData(inputs, outputs, inputs, outputs);
    }

    /**
     * @param inputs                   Array of input matrices.
     * @param outputs                  Array of output matrices.
     * @param percentageOfTrainingData Percentage of training data selected from the given data.
     * @return Object that encapsulates the separated data.
     * @throws IllegalArgumentException if inputs and outputs array lengths mismatch.
     */
    public static SeparatedData separateData(IMatrix[] inputs, IMatrix[] outputs, double percentageOfTrainingData) {
        if (inputs.length != outputs.length)
            throw new IllegalArgumentException("Inputs and outputs length don't match: " + inputs.length + " != " + outputs.length);

        // Separated arrays.
        ArrayList<IMatrix> trainingInputs = new ArrayList<>();
        ArrayList<IMatrix> trainingOutputs = new ArrayList<>();
        ArrayList<IMatrix> testInputs = new ArrayList<>();
        ArrayList<IMatrix> testOutputs = new ArrayList<>();

        // Equidistant data selection
        double accumulator = 0;
        for (int i = 0; i < inputs.length; i++) {
            accumulator += percentageOfTrainingData;
            if (accumulator > 1) {
                trainingInputs.add(inputs[i]);
                trainingOutputs.add(outputs[i]);
                accumulator -= 1;
            } else {
                testInputs.add(inputs[i]);
                testOutputs.add(outputs[i]);
            }
        }

        // Encapsulation of sets.
        return new SeparatedData(
                trainingInputs.toArray(new IMatrix[]{}),
                trainingOutputs.toArray(new IMatrix[]{}),
                testInputs.toArray(new IMatrix[]{}),
                testOutputs.toArray(new IMatrix[]{})
        );
    }

    public static SeparatedData[] toBatch(SeparatedData data) {
        return new SeparatedData[]{data};
    }

    // FIXME Not working for test sets different from training sets.
    public static SeparatedData[] toStocastic(SeparatedData data) {
        SeparatedData[] batch = new SeparatedData[data.getTrainingInputs().length];

        for (int i = 0; i < batch.length; i++)
            batch[i] = DataSeparator.simpleData(new IMatrix[]{data.getTrainingInputs()[i]}, new IMatrix[]{data.getTrainingOutputs()[i]});

        return batch;
    }

    // FIXME Not working for test sets different from training sets.
    public static SeparatedData[] toMiniBatch(SeparatedData data, int batchSize, boolean shuffle) {
        int size = data.getTrainingInputs().length;

        if(shuffle)
        data.shuffleData();

        SeparatedData[] dataMB = new SeparatedData[(int) Math.ceil(size / (double) batchSize)];
        for (int i = 0; i < size; ) {
            ArrayList<IMatrix> batchI = new ArrayList<>();
            ArrayList<IMatrix> batchO = new ArrayList<>();
            for (int j = 0; j < batchSize && i < size; j++, i++) {
                batchI.add(data.getTrainingInputs()[i]);
                batchO.add(data.getTrainingOutputs()[i]);
            }
            dataMB[(i - 1) / batchSize] = DataSeparator.simpleData(batchI.toArray(new IMatrix[]{}), batchO.toArray(new IMatrix[]{}));
        }

        return dataMB;
    }
}
