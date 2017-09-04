package com.lirfu.networks;

import com.lirfu.graphicslib.matrix.IMatrix;

import java.util.ArrayList;

/**
 * Created by lirfu on 26.08.17..<br>
 * Class implements separation of input-output data pairs into the training and test sets of data pairs.
 */
public class DataSeparator {
    /**
     *
     * @param inputs Array of input matrices.
     * @param outputs Array of output matrices.
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
}
