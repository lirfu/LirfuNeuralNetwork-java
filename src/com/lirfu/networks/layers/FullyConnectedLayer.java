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
    private DescendMethod mWeightsDescendMethod;
    private DescendMethod mBiasDescendMethod;


    public FullyConnectedLayer(int inputSize, int outputSize, DerivativeFunction function, DescendMethod descendMethod, WeightInitializer initializer) {
        super(inputSize, outputSize, function, initializer);
        mWeightsDescendMethod = descendMethod.copy();
        mBiasDescendMethod = descendMethod.copy();
    }

    private FullyConnectedLayer(FullyConnectedLayer fullyConnectedLayer) {
        super(fullyConnectedLayer);
        mWeightsDescendMethod = fullyConnectedLayer.mWeightsDescendMethod.copy();
        mBiasDescendMethod = fullyConnectedLayer.mBiasDescendMethod.copy();
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

    @Override
    public Layer copy() {
        return new FullyConnectedLayer(this);
    }
}
