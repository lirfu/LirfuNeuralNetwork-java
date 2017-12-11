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
     *
     * @param inputSize  Number of expected inputs (for weights).
     * @param outputSize Number of this layer's outputs (number of neurons).
     * @param function   The layer's activation function.
     */
    protected InnerLayer(int inputSize, int outputSize, DerivativeFunction function) {
        super(new Matrix(1, outputSize));

        this.function = function;
    }

    /**
     * Method for fetching the next random double.
     *
     * @param rand The instance of Random.
     * @param min  The range minimum.
     * @param max  The range maximum.
     * @return The next random value from the given range.
     */
    protected double nextRandom(Random rand, double min, double max) {
        return rand.nextDouble() * (max - min) + min;
    }

    /**
     * Getter for the weight matrix of this layer.
     * @return The matrix of weights.
     */
    public IMatrix getWeights() {
        return weights;
    }

    /**
     * Getter for this layer's activation function.
     *
     * @return The activation function.
     */
    public DerivativeFunction getFunction() {
        return function;
    }

    /**
     * Takes the output values from the previous layer and both calculates and stores this layer's outputs.
     *
     * @param leftLayer The previous (left) layer.
     */
    public abstract void forwardPass(Layer leftLayer);

    /**
     * Updates the weight values
     *
     * @param outDiff      Errors from the last layer or total error if this is the output layer.
     * @param leftOutputs  Outputs from the last layer.
     * @param learningRate The learning rate.
     * @return Processed errors from the last layer (to be used by the left layer).
     */
    public abstract IMatrix backwardPass(IMatrix outDiff, IMatrix leftOutputs, double learningRate);
}
