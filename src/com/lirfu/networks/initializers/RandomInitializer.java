package com.lirfu.networks.initializers;

import com.lirfu.graphicslib.matrix.IMatrix;

import java.util.Random;

public class RandomInitializer implements WeightInitializer {
    private double mMin;
    private double mMax;

    private Random rand = new Random();

    public RandomInitializer(double min, double max) {
        mMin = min;
        mMax = max;
    }

    @Override
    public void initialize(IMatrix weights) {
        for (int c = 0; c < weights.getColsCount(); c++)
            for (int r = 0; r < weights.getRowsCount(); r++)
                weights.set(r, c, nextRandom(rand, mMin, mMax));
    }

    @Override
    public void initialize(IMatrix biases, IMatrix weights) {
        for (int c = 0; c < weights.getColsCount(); c++) {
            for (int r = 0; r < weights.getRowsCount(); r++)
                weights.set(r, c, nextRandom(rand, mMin, mMax));

            biases.set(0, c, nextRandom(rand, mMin, mMax));
        }
    }

    /**
     * Method for fetching the next random double.
     *
     * @param rand The instance of Random.
     * @param min  The range minimum.
     * @param max  The range maximum.
     * @return The next random value from the given range.
     */
    private double nextRandom(Random rand, double min, double max) {
        return rand.nextDouble() * (max - min) + min;
    }
}
