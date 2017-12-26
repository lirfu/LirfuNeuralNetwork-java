package com.lirfu.networks.descendmethods;

import com.lirfu.graphicslib.matrix.IMatrix;

public class MomentumDescend implements DescendMethod {
    private double momentumDecay;

    private IMatrix previousVelocity;

    public MomentumDescend(double momentumDecay) {
        this.momentumDecay = momentumDecay;
    }

    private MomentumDescend(MomentumDescend momentumDescend){
        momentumDecay = momentumDescend.momentumDecay;
        previousVelocity = momentumDescend.previousVelocity;
    }

    @Override
    public void performDescend(IMatrix previousWeights, IMatrix gradient) {
        if (previousVelocity == null)
            previousVelocity = gradient;
        else
            previousVelocity
                    .scalarMultiply(momentumDecay)
                    .add(gradient);

        previousWeights.sub(previousVelocity);
    }

    @Override
    public DescendMethod copy() {
        return new MomentumDescend(this);
    }
}
