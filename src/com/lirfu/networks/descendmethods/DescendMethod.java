package com.lirfu.networks.descendmethods;

import com.lirfu.graphicslib.matrix.IMatrix;

public interface DescendMethod {
    public void performDescend(IMatrix previousWeights, IMatrix gradient);
    public DescendMethod copy();
}
