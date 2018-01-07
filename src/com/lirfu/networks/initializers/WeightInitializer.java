package com.lirfu.networks.initializers;

import com.lirfu.graphicslib.matrix.IMatrix;

public interface WeightInitializer {
    /**
     * Initializes values of given weights matrix.
     * @param weights Matrix to initialize.
     */
    public void initialize(IMatrix weights);
    /**
     * Initializes values of given biases and weights matrix. Uses same loop to iterate both.
     * @param biases Matrix to initialize.
     * @param weights Matrix to initialize.
     */
    public void initialize(IMatrix biases, IMatrix weights);
}
