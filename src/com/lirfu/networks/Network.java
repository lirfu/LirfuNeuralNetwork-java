package com.lirfu.networks;

import com.lirfu.graphicslib.IncompatibleOperandException;
import com.lirfu.graphicslib.matrix.IMatrix;
import com.lirfu.graphicslib.matrix.Matrix;
import com.lirfu.networks.layers.InnerLayer;
import com.lirfu.networks.layers.InputLayer;
import com.lirfu.networks.layers.Layer;
import org.omg.PortableServer.IMPLICIT_ACTIVATION_POLICY_ID;

import java.util.Arrays;

/**
 * Created by lirfu on 08.08.17..
 */
public class Network {
    private InputLayer inputLayer;
    private InnerLayer[] hiddenLayers;

    public Network(InputLayer inputLayer, InnerLayer... hiddenLayers) {
        this.hiddenLayers = hiddenLayers;
        this.inputLayer = inputLayer;
    }

    public Network(Network network) {
        inputLayer = (InputLayer) network.inputLayer.copy();
        hiddenLayers = new InnerLayer[network.hiddenLayers.length];
        for (int i = 0; i < hiddenLayers.length; i++)
            hiddenLayers[i] = (InnerLayer) network.hiddenLayers[i].copy();
    }

    /**
     * Calculates the output of the neural network for the given input.
     *
     * @param input Matrix of inputs.
     * @return Matrix of outputs.
     * @throws IncompatibleOperandException if input-weights matrix dimensions between layers mismatch.
     */
    public IMatrix getOutput(IMatrix input) throws IncompatibleOperandException {
        Layer lastLayer = inputLayer;
        ((InputLayer) lastLayer).setOutput(input);

        InnerLayer currentLayer;
        for (InnerLayer hiddenLayer : hiddenLayers) {
            currentLayer = hiddenLayer;
            currentLayer.forwardPass(lastLayer);
            lastLayer = currentLayer;
        }

        return lastLayer.getOutput();
    }

    public double backpropagate(double learningRate, SeparatedData[] dataBatches) throws IncompatibleOperandException {
        double error = 0;
        // Updating for each batch.
        for (SeparatedData batch : dataBatches) {
            // Accumulating deltas for each sample in batch.
            for (int i = 0; i < batch.getTrainingInputs().length; i++) {
                IMatrix input = batch.getTrainingInputs()[i];
                IMatrix targetOutput = batch.getTrainingOutputs()[i];
                IMatrix outDiff = new Matrix(targetOutput.getDimension());

                // Forward pass to get output
                IMatrix guessedOutput = getOutput(input);

                // Calculate the output difference
                outDiff.add(guessedOutput.nSub(targetOutput));

                outDiff = outDiff.nTranspose(false); // FIXME .scalarMultiply(1. / batch.getTrainingInputs().length)

                // Iterate through the rest of the layers (backwards)
                InnerLayer currentLayer;
                Layer leftLayer;
                for (int l = hiddenLayers.length - 1; l >= 0; l--) {
                    currentLayer = hiddenLayers[l];
                    if (l == 0)
                        leftLayer = inputLayer;
                    else
                        leftLayer = hiddenLayers[l - 1];

//                // Layer differences
//                differences = outDiff.nHadamardProduct(currentLayer.getNet().nApplyFunction(currentLayer.getFunction().getDerivative()).nTranspose(false));
//
//                // Update the differences for the next iteration
//                outDiff = currentLayer.getWeights().nMultiply(differences);

                    // Update weights
//                currentLayer.getWeights().add(differences.nMultiply(leftLayer.getOutput()).scalarMultiply(learningRate).nTranspose(false));
//                currentLayer.getBiases().add(differences.scalarMultiply(learningRate).nTranspose(false));

                    // Accumulate deltas throughout the network.
                    outDiff = currentLayer.backwardPass(outDiff, leftLayer.getOutput(), learningRate);
                }
            }

            // Update weights with accumulated deltas.
            for (InnerLayer layer : hiddenLayers)
                layer.updateWeights();

            // Use test inputs to calculate the final error.
            error += calculateError(batch.getTestInputs(), batch.getTestOutputs());
        }
        return error / dataBatches.length; // Return normalized error
    }

    public double calculateError(IMatrix[] inputs, IMatrix[] outputs) {
        double totalError = 0;
        for (int i = 0; i < inputs.length; i++) {
            // Forward pass to get output
            IMatrix guessedOutput = getOutput(inputs[i]);

            // Calculate the output difference
            IMatrix outDiff = outputs[i].nSub(guessedOutput).nTranspose(false);

            // Calculate the total output error
            for (int r = 0; r < outDiff.getRowsCount(); r++)
                for (int c = 0; c < outDiff.getColsCount(); c++)
                    totalError += outDiff.get(r, c) * outDiff.get(r, c);
        }
        return totalError * 0.5 / inputs.length;
    }

    @Override
    public String toString() {
        String s = "";

        for (int i = 0; i < hiddenLayers.length; i++)
            s += "\tLayer " + i + ":\n" + hiddenLayers[i].getWeights().toString(4) + "\n";

        return s;
    }

}
