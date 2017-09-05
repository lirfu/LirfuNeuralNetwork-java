package com.lirfu.networks.layers;

import com.lirfu.graphicslib.matrix.IMatrix;
import com.lirfu.graphicslib.matrix.MatrixDimension;

/**
 * Created by lirfu on 08.08.17..<br>
 * Layer represents an abstract layer that is used to build a neural network.<br>
 *     It stores the net inputs of the neurons in the layer and those neuron's outputs.
 */
public abstract class Layer{
    protected IMatrix net;
    protected IMatrix output;

    /**
     * Constructor for an abstract layer.
     * @param output Matrix containing this layer's outputs (dimensions describe the layer's neuron structure).
     */
    protected Layer(IMatrix output) {
        this.output = output;
    }

    /**
     * Getter for the last calculated net inputs for layer's neurons.
     * @return Matrix containing the neuron nets (dimensions describe the layer's neuron structure).
     */
    public IMatrix getNet() {
        return net;
    }

    /**
     * Getter for the last calculated activated outputs for this layer's neurons.
     * @return Matrix containing the activated neuron outputs (dimensions describe the layer's neuron structure).
     */
    public IMatrix getOutput() {
        return output;
    }

    /**
     * Getter for the output dimension (describes this layer's neuron structure).
     * @return
     */
    public MatrixDimension getOutputDimension(){
        return output.getDimension();
    }

}
