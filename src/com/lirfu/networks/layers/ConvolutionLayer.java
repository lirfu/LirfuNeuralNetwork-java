package com.lirfu.networks.layers;

import com.lirfu.graphicslib.IncompatibleOperandException;
import com.lirfu.graphicslib.functions.DerivativeFunction;
import com.lirfu.graphicslib.matrix.IMatrix;
import com.lirfu.graphicslib.matrix.Matrix;
import com.lirfu.graphicslib.matrix.MatrixDimension;
import com.lirfu.graphicslib.matrix.MatrixUtils;

/**
 * Created by lirfu on 25.08.17..
 */
public class ConvolutionLayer extends InnerLayer {
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

        this.weights = new Matrix(templateSize);
        this.inputSize = inputSize;
    }

    @Override
    public void forwardPass(Layer leftLayer) {
        MatrixDimension d = leftLayer.getOutputDimension();

        Matrix output = new Matrix(d.rows() - weights.getRowsCount() + 1, d.cols() - weights.getColsCount() + 1);
        Matrix nets = new Matrix(output.getDimension());

        // Iterate through top-left anchor indexes
        for (int r = 0; r < output.getRowsCount(); r++) {
            for (int c = 0; c < output.getColsCount(); c++) {
                double net = MatrixUtils.sumValues(
                        leftLayer.getOutput()
                                .subMatrix(r, r + weights.getRowsCount() - 1, c, c + weights.getColsCount() - 1, false)
                                .HadamardProduct(weights)
                );

                // Set this neuron's net and output
                nets.set(r, c, net);
            }
        }

        output = (Matrix) nets.nApplyFunction(function.getFunction());

        this.net = nets;
        this.output = output;
    }

    @Override
    public IMatrix backwardPass(IMatrix outputDifferences, IMatrix leftOutputs, double learningRate) {
        // Layer differences
        IMatrix differences = outputDifferences.nHadamardProduct(net.nApplyFunction(function.getDerivative()).nTranspose(false));

        // Update the differences for the next iteration
        outputDifferences = new Matrix(inputSize);

        // Iterate through inputs - weights size
        for (int r = 0; r < outputDifferences.getRowsCount() - weights.getRowsCount() + 1; r++) {
            for (int c = 0; c < outputDifferences.getColsCount() - weights.getColsCount() + 1; c++) {

                for (int y = 0; y < weights.getRowsCount(); y++) {
                    for (int x = 0; x < weights.getColsCount(); x++) {

                        outputDifferences.set(
                                r + y, c + x,
                                outputDifferences.get(r + y, c + x) + weights.get(y, x) * differences.get(r, c)
                        );

                        newTemlate.set(
                                y, x,
                                newTemlate.get(y, x) + differences.get(r, c) * leftOutputs.get()
                        );

                    }
                }

            }
        }

        IMatrix newTemlate = new Matrix(weights.getDimension());


        // Update weight and bias values
        weights.add(differences.nMultiply(leftOutputs).scalarMultiply(learningRate).nTranspose(false));
        biases.add(differences.scalarMultiply(learningRate).nTranspose(false));

        return outputDifferences;
    }
}
