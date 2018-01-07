package com.lirfu.networks.layers;

import com.lirfu.graphicslib.functions.DerivativeFunction;
import com.lirfu.graphicslib.matrix.IMatrix;
import com.lirfu.graphicslib.matrix.Matrix;
import com.lirfu.networks.descendmethods.DescendMethod;
import com.lirfu.networks.initializers.WeightInitializer;

/**
 * Created by lirfu on 08.08.17..
 */
public class FullyConnectedLayer extends InnerLayer {
    protected int inputSize;
    protected DerivativeFunction function;

    protected IMatrix net;
    protected IMatrix weights;
    protected IMatrix biases;
    protected IMatrix weightDeltas;
    protected IMatrix biasDeltas;

    private DescendMethod mWeightsDescendMethod;
    private DescendMethod mBiasDescendMethod;


    public FullyConnectedLayer(int inputSize, int outputSize, DerivativeFunction function, DescendMethod descendMethod, WeightInitializer initializer) {
        super(new Matrix(1, outputSize));

        this.inputSize = inputSize;
        this.function = function;

        biases = new Matrix(1, outputSize);
        weights = new Matrix(inputSize, outputSize);

        biasDeltas = new Matrix(1, outputSize);
        weightDeltas = new Matrix(inputSize, outputSize);

        initializer.initialize(biases, weights);

        mWeightsDescendMethod = descendMethod.copy();
        mBiasDescendMethod = descendMethod.copy();
    }

    private FullyConnectedLayer(FullyConnectedLayer fullyConnectedLayer) {
        super(fullyConnectedLayer);
        inputSize = fullyConnectedLayer.inputSize;
        function = fullyConnectedLayer.function;
        weights = fullyConnectedLayer.weights.copy();
        biases = fullyConnectedLayer.biases.copy();
        weightDeltas = fullyConnectedLayer.weightDeltas.copy();
        biasDeltas = fullyConnectedLayer.biasDeltas.copy();

        mWeightsDescendMethod = fullyConnectedLayer.mWeightsDescendMethod.copy();
        mBiasDescendMethod = fullyConnectedLayer.mBiasDescendMethod.copy();

        if (fullyConnectedLayer.net != null)
            net = fullyConnectedLayer.net;
    }

    public void forwardPass(Layer leftLayer) {
        net = leftLayer.getOutput().nMultiply(weights).add(biases);
        output = net.nApplyFunction(function.getFunction());
    }

    public IMatrix backwardPass(IMatrix outputDifferences, IMatrix leftOutputs, double learningRate) {
        // Layer differences
        IMatrix differences = outputDifferences.nHadamardProduct(net.nApplyFunction(function.getDerivative()).nTranspose(false));

        // Update the differences for the next iteration
        outputDifferences = weights.nMultiply(differences);

        // Accumulate weight and bias deltas.
        mWeightsDescendMethod.performDescend(weightDeltas, differences.nMultiply(leftOutputs).scalarMultiply(learningRate).nTranspose(false));
        mBiasDescendMethod.performDescend(biasDeltas, differences.scalarMultiply(learningRate).nTranspose(false));

        return outputDifferences;
    }

    @Override
    public void updateWeights() {
        // Update weights.
        weights.add(weightDeltas);
        biases.add(biasDeltas);

        // Reset deltas.
        weightDeltas = new Matrix(weightDeltas.getDimension());
        biasDeltas = new Matrix(biasDeltas.getDimension());
    }


    /**
     * Getter for the last calculated net inputs for layer's neurons.
     *
     * @return Matrix containing the neuron nets (dimensions describe the layer's neuron structure).
     */
    public IMatrix getNet() {
        return net;
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

    @Override
    public Layer copy() {
        return new FullyConnectedLayer(this);
    }

    @Override
    public String toString() {
        return biases.toString(2) + '\n' + weights.toString(2);
    }
}
