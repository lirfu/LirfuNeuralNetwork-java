package com.lirfu.networks.layers;

import com.lirfu.graphicslib.matrix.IMatrix;
import com.lirfu.graphicslib.matrix.Matrix;
import com.lirfu.graphicslib.matrix.MatrixDimension;

/**
 * Created by lirfu on 08.08.17..
 */
public class InputLayer extends Layer {
    /**
     * Constructor for a 1D input layer.
     *
     * @param outputSize Number of input neurons.
     */
    public InputLayer(int outputSize) {
        super(new Matrix(1, outputSize));
    }

    private InputLayer(InputLayer inputLayer) {
        super(inputLayer);
    }

    /**
     * Constructor for a 2D input layer.
     *
     * @param outputSize Dimensions of the input neuron array.
     */
    public InputLayer(MatrixDimension outputSize) {
        super(new Matrix(outputSize));
    }

    /**
     * Sets the output values of this input layer (in other words sets data on the input layer)
     *
     * @param output Matrix
     * @throws IllegalArgumentException if the given matrix dimensions mismatch the dimensions set in the constructor (layer dimension is constant).
     */
    public void setOutput(IMatrix output) {
        if (!this.output.getDimension().equals(output.getDimension()))
            throw new IllegalArgumentException("Matrices are not the same size: " + this.output.getDimension() + " != " + output.getDimension());

        this.output = output;
    }

    @Override
    public Layer copy() {
        return new InputLayer(this);
    }
}
