package com.lirfu.networks.initializers;

import com.lirfu.graphicslib.matrix.IMatrix;

public interface WeightInitializer {
    public void initialize(IMatrix biases, IMatrix weights);
}
