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


    public FullyConnectedLayer(int inputSize, int neuronNumber, DerivativeFunction function, DescendMethod descendMethod, WeightInitializer initializer) {
        super(new Matrix(1, neuronNumber));

        this.inputSize = inputSize;
        this.function = function;

        biases = new Matrix(1, neuronNumber);
        weights = new Matrix(inputSize, neuronNumber);

        biasDeltas = new Matrix(1, neuronNumber);
        weightDeltas = new Matrix(inputSize, neuronNumber);

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
    public int numberOfParameters() {
        return biases.getRowsCount() * biases.getColsCount() + weights.getRowsCount() * weights.getColsCount();
    }

    @Override
    public Layer copy() {
        return new FullyConnectedLayer(this);
    }

    @Override
    public String toString() {
        return biases.toString(2) + '\n' + weights.toString(2);
    }

    @Override
    public double[] getNeuron(int index) {
        double[] array = new double[weights.getRowsCount() + 1];

        array[0] = biases.get(0, index);
        for (int r = 0; r < weights.getRowsCount(); r++)
            array[r + 1] = weights.get(r, index);

        return array;
    }

    @Override
    public void setNeuron(int index, double[] values) {
        biases.set(0, index, values[0]);

        for (int r = 0; r < weights.getRowsCount(); r++)
            weights.set(r, index, values[r + 1]);
    }
}
