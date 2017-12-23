package com.lirfu.networks.layers;

import com.lirfu.graphicslib.functions.DerivativeFunction;
import com.lirfu.graphicslib.matrix.IMatrix;
import com.lirfu.networks.initializers.WeightInitializer;

/**
 * Created by lirfu on 08.08.17..
 */
public class FullyConnectedLayer extends InnerLayer {
    public FullyConnectedLayer(int inputSize, int outputSize, DerivativeFunction function, WeightInitializer initializer) {
        super(inputSize, outputSize, function, initializer);
    }

    private FullyConnectedLayer(FullyConnectedLayer fullyConnectedLayer) {
        super(fullyConnectedLayer);
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

        // Update weight and bias values
        weights.add(differences.nMultiply(leftOutputs).scalarMultiply(learningRate).nTranspose(false));
        biases.add(differences.scalarMultiply(learningRate).nTranspose(false));

        return outputDifferences;
    }

    @Override
    public Layer copy() {
        return new FullyConnectedLayer(this);
    }
}
