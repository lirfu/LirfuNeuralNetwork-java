package com.lirfu.networks.layers;

import com.lirfu.graphicslib.functions.DerivativeFunction;
import com.lirfu.graphicslib.matrix.IMatrix;

/**
 * Created by lirfu on 25.08.17..
 */
public class ConvolutionalLayer extends InnerLayer {
    protected ConvolutionalLayer(int inputSize, int outputSize, DerivativeFunction function) {
        super(inputSize, outputSize, function);
    }

    @Override
    public void forwardPass(Layer leftLayer) {

    }

    @Override
    public void updateWeights(IMatrix differences, IMatrix leftOutputs, double learningRate) {

    }
}
