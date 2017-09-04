package com.lirfu.networks.layers;

import com.lirfu.graphicslib.functions.DerivativeFunction;
import com.lirfu.graphicslib.matrix.IMatrix;
import com.lirfu.graphicslib.matrix.Matrix;

import java.util.Random;

/**
 * Created by lirfu on 24.08.17..<br>
 * Layer represents an abstract inner layer that is used to build a neural network.<br>
 * It stores the layer's weights, bias values and the activation function.
 */
public abstract class InnerLayer extends Layer {
    protected DerivativeFunction function;
    protected IMatrix weights;
    protected IMatrix biases;

    /**
     * Constructor for an abstract inner (or output) layer.
     * @param inputSize Number of expected inputs (for weights).
     * @param outputSize Number of this layer's outputs (number of neurons).
     * @param function The layer's activation function.
     */
    protected InnerLayer(int inputSize, int outputSize, DerivativeFunction function) {
        super(new Matrix(1, outputSize));

        this.function = function;
        this.weights = new Matrix(inputSize, outputSize);
        this.biases = new Matrix(1, outputSize);

        // Init weight values
        Random rand = new Random();
        for (int c = 0; c < weights.getColsCount(); c++) {
            for (int r = 0; r < weights.getRowsCount(); r++)
                weights.set(r, c, nextRandom(rand));

            biases.set(0, c, nextRandom(rand));
        }
    }

    /**
     * Method for fetching the next random double.
     * @param rand The instance of Random.
     * @return The next random value from [-1, 1].
     */
    private double nextRandom(Random rand) {
        return rand.nextDouble() * 2 - 1;
    }

    /**
     * Getter for the weight matrix of this layer.
     * @return The matrix of weights.
     */
    public IMatrix getWeights() {
        return weights;
    }

    /**
     * Getter for the bias values of this layer.
     * @return The matrix of bias values (a row matrix).
     */
    public IMatrix getBiases() {
        return biases;
    }

    /**
     * Getter for this layer's activation function.
     * @return The activation function.
     */
    public DerivativeFunction getFunction() {
        return function;
    }

    /**
     * Takes the output values from the previous layer and both calculates and stores this layer's outputs.
     * @param leftLayer The previous (left) layer.
     */
    public abstract void forwardPass(Layer leftLayer);

    /**
     * Updates the weight values
     * @param differences Activated differences of this layer.
     * @param leftOutputs Outputs from the last layer.
     * @param learningRate The learning rate.
     */
    public abstract void updateWeights(IMatrix differences, IMatrix leftOutputs, double learningRate);
}
