package com.lirfu.networks.layers;

import com.lirfu.graphicslib.matrix.IMatrix;

/**
 * Created by lirfu on 24.08.17..<br>
 * Layer represents an abstract inner layer that is used to build a neural network.<br>
 * It stores the layer's weights, bias values and the activation function.
 */
public abstract class InnerLayer extends Layer {
    protected InnerLayer(IMatrix output) {
        super(output);
    }

    protected InnerLayer(Layer layer) {
        super(layer);
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
