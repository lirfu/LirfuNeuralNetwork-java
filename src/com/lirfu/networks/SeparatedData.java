package com.lirfu.networks;

import com.lirfu.graphicslib.matrix.IMatrix;
import com.sun.istack.internal.NotNull;
import com.sun.istack.internal.Nullable;

import java.util.Random;

/**
 * Created by lirfu on 03.09.17..<br>
 * SeparatedData encapsulates the data needed to train the neural network.<br>
 * The data is split into a training and test set which are then accordingly used by the neural network for training and validation.
 */
public class SeparatedData {
    private IMatrix[] trainingInputs;
    private IMatrix[] trainingOutputs;
    private IMatrix[] testInputs;
    private IMatrix[] testOutputs;

    /**
     * Constructor for encapsulating individual sets of training and test input-output pairs.
     *
     * @param trainingInputs  Array of input matrices for training.
     * @param trainingOutputs Array of output matrices for training.
     * @param testInputs      Array of input matrices for testing.
     * @param testOutputs     Array of output matrices for testing.
     * @throws IllegalArgumentException if array length of inputs and outputs arrays mismatch for training and testing sets accordingly.
     */
    public SeparatedData(@NotNull IMatrix[] trainingInputs, @NotNull IMatrix[] trainingOutputs, @NotNull IMatrix[] testInputs, @NotNull IMatrix[] testOutputs) {
        if (trainingInputs.length != trainingOutputs.length)
            throw new IllegalArgumentException("Training data array lengths don't match: " + trainingInputs.length + " != " + trainingOutputs.length);
        if (testInputs.length != testOutputs.length)
            throw new IllegalArgumentException("Test data array lengths don't match: " + testInputs.length + " != " + testOutputs.length);


        this.trainingInputs = trainingInputs;
        this.trainingOutputs = trainingOutputs;

        this.testInputs = testInputs;
        this.testOutputs = testOutputs;
    }

    public void shuffleData() {
        Random rand = new Random();
        for (int i = 0; i < trainingInputs.length && i < testInputs.length; i++)
            if (rand.nextBoolean()) {
                int randIndexTr = rand.nextInt(trainingInputs.length);
                int randIndexTs = rand.nextInt(testInputs.length);

                IMatrix tI = trainingInputs[randIndexTr];
                trainingInputs[randIndexTr] = testInputs[randIndexTs];
                testInputs[randIndexTs] = tI;

                IMatrix tO = trainingOutputs[randIndexTr];
                trainingOutputs[randIndexTr] = testOutputs[randIndexTs];
                testOutputs[randIndexTs] = tO;

            }
    }

    public IMatrix[] getTrainingInputs() {
        return trainingInputs;
    }

    public IMatrix[] getTrainingOutputs() {
        return trainingOutputs;
    }

    public IMatrix[] getTestInputs() {
        return testInputs;
    }

    public IMatrix[] getTestOutputs() {
        return testOutputs;
    }
}