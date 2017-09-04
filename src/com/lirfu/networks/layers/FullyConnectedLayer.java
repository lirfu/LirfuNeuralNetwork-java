package com.lirfu.networks.layers;

import com.lirfu.graphicslib.functions.DerivativeFunction;
import com.lirfu.graphicslib.matrix.IMatrix;

/**
 * Created by lirfu on 08.08.17..
 */
public class FullyConnectedLayer extends InnerLayer {
    public FullyConnectedLayer(int inputSize, int outputSize, DerivativeFunction function) {
        super(inputSize, outputSize, function);
    }

    public void forwardPass(Layer leftLayer) {
        net = leftLayer.getOutput().nMultiply(weights).add(biases);
        output = net.nApplyFunction(function.getFunction());
    }

    public void updateWeights(IMatrix differences, IMatrix leftOutputs, double learningRate){
        weights.add(differences.nMultiply(leftOutputs).scalarMultiply(learningRate).nTranspose(false));
    }
}
