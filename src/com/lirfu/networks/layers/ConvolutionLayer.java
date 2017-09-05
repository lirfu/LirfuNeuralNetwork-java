package com.lirfu.networks.layers;

import com.lirfu.graphicslib.functions.DerivativeFunction;
import com.lirfu.graphicslib.matrix.IMatrix;
import com.lirfu.graphicslib.matrix.Matrix;
import com.lirfu.graphicslib.matrix.MatrixDimension;

/**
 * Created by lirfu on 25.08.17..
 */
public class ConvolutionLayer extends InnerLayer {
    private IMatrix template;
    private MatrixDimension inputSize;

    /**
     * Constructor for a convolution layer.
     *
     * @param inputSize    Size of the 2D input array.
     * @param templateSize Size of the template square used for convolution.
     * @param function     The layer's activation function.
     */
    public ConvolutionLayer(MatrixDimension inputSize, MatrixDimension templateSize, DerivativeFunction function) {
        super(templateSize.rows(), templateSize.cols(), function);

        if (inputSize.rows() < templateSize.rows() || inputSize.cols() < templateSize.cols())
            throw new IllegalArgumentException("Template size is too large for input size: " + templateSize + " >> " + inputSize);

        this.template = new Matrix(templateSize);
        this.inputSize = inputSize;
    }

    @Override
    public void forwardPass(Layer leftLayer) {
        // Temp matrix to hold structured inputs.
        Matrix input = new Matrix(inputSize);

        // Copy results into a matrix.
        for (int r = 0; r < inputSize.rows(); r++)
            for (int c = 0; c < inputSize.cols(); c++) {
                input.set(
                        r, c,
                        leftLayer.getOutput().get(1, r * inputSize.cols() + c)
                );
            }

        // TODO ...
    }

    @Override
    public void updateWeights(IMatrix differences, IMatrix leftOutputs, double learningRate) {
// TODO ???
    }
}
