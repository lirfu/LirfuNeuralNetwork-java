package com.lirfu.networks.layers;

import com.lirfu.graphicslib.functions.DerivativeFunction;
import com.lirfu.graphicslib.matrix.IMatrix;
import com.lirfu.graphicslib.matrix.Matrix;
import com.lirfu.networks.initializers.WeightInitializer;

/**
 * Created by lirfu on 24.08.17..<br>
 * Layer represents an abstract inner layer that is used to build a neural network.<br>
 * It stores the layer's weights, bias values and the activation function.
 */
public abstract class InnerLayer extends Layer {
    protected int inputSize;
    protected DerivativeFunction function;
    protected IMatrix weights;
    protected IMatrix biases;
    protected IMatrix weightDeltas;
    protected IMatrix biasDeltas;

    /**
     * Constructor for an abstract inner (or output) layer.
     *
     * @param inputSize  Number of expected inputs (for weights).
     * @param outputSize Number of this layer's outputs (number of neurons).
     * @param function   The layer's activation function.
     */
    protected InnerLayer(int inputSize, int outputSize, DerivativeFunction function, WeightInitializer initializer) {
        super(new Matrix(1, outputSize));

        this.inputSize = inputSize;
        this.function = function;

        biases = new Matrix(1, outputSize);
        weights = new Matrix(inputSize, outputSize);

        biasDeltas = new Matrix(1, outputSize);
        weightDeltas = new Matrix(inputSize, outputSize);

        initializer.initialize(biases, weights);
    }

    protected InnerLayer(InnerLayer innerLayer) {
        super(innerLayer);
        inputSize = innerLayer.inputSize;
        function = innerLayer.function;
        weights = innerLayer.weights.copy();
        biases = innerLayer.biases.copy();
        weightDeltas = innerLayer.weightDeltas.copy();
        biasDeltas = innerLayer.biasDeltas.copy();
    }

    /**
     * Getter for the weight matrix of this layer.
     *
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
     * Accumulates the weight changes (deltas).
     *
     * @param outDiff      Errors from the last layer or total error if this is the output layer.
     * @param leftOutputs  Outputs from the last layer.
     * @param learningRate The learning rate.
     * @return Processed errors from the last layer (to be used by the left layer).
     */
    public abstract IMatrix backwardPass(IMatrix outDiff, IMatrix leftOutputs, double learningRate);

    /**
     * Updates the weights with accumulated deltas. Resets the deltas.
     */
    public abstract void updateWeights();
}
