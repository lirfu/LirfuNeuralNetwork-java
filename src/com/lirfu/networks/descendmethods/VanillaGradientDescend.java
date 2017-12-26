package com.lirfu.networks.descendmethods;

import com.lirfu.graphicslib.matrix.IMatrix;

public class VanillaGradientDescend implements DescendMethod {
    @Override
    public void performDescend(IMatrix previousWeights, IMatrix gradient) {
        previousWeights.sub(gradient);
    }

    @Override
    public DescendMethod copy() {
        return new VanillaGradientDescend();
    }
}
