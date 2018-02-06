//package com.lirfu.networks.demo;
//
//import com.lirfu.graphicslib.IncompatibleOperandException;
//import com.lirfu.graphicslib.functions.DerivativeFunction;
//import com.lirfu.graphicslib.functions.Sigmoid;
//import com.lirfu.graphicslib.matrix.IMatrix;
//import com.lirfu.graphicslib.matrix.Matrix;
//import com.lirfu.graphicslib.matrix.MatrixDimension;
//import com.lirfu.graphicslib.vector.Vector;
//import com.lirfu.lirfugraph.graphs.LinearGraph;
//import com.lirfu.lirfugraph.Row;
//import com.lirfu.lirfugraph.VerticalContainer;
//import com.lirfu.lirfugraph.Window;
//import com.lirfu.networks.Network;
//import com.lirfu.networks.datatransformations.BinaryImage;
//import com.lirfu.networks.datatransformations.MalformedFileException;
//import com.lirfu.networks.layers.FullyConnectedLayer;
//import com.lirfu.networks.layers.InputLayer;
//
//import java.io.File;
//
///**
// * Created by lirfu on 05.09.17..
// */
//public class ImageClassifierDemo {
//    public static void main(String[] args) throws MalformedFileException, IncompatibleOperandException {
//        File resourcesDir = new File("res/BinaryImages");
//        File[] inputFiles = resourcesDir.listFiles();
//
//        IMatrix[] inputs = new IMatrix[inputFiles.length];
//        IMatrix[] outputs = new IMatrix[inputFiles.length];
//
//        MatrixDimension dimension = null;
//        for (int i = 0; i < inputFiles.length; i++) {
//            inputs[i] = new BinaryImage(inputFiles[i]).toMatrix();
//            outputs[i] = new Matrix(new Vector(inputFiles[i].getName().toCharArray()[0]));
//
//            if (dimension == null)
//                dimension = inputs[i].getDimension();
//            if (!dimension.equals(inputs[i].getDimension()))
//                throw new IncompatibleOperandException("File " + inputFiles[i].getName() + " doesn't match dimensions " + dimension);
//        }
//
//        MatrixDimension templateSize = new MatrixDimension(3, 3);
//        DerivativeFunction activation = new Sigmoid();
//        Network net = new Network(
//                1e-2,
//                new InputLayer(dimension),
//                new ConvolutionLayer(dimension, templateSize, activation),
//                new ConvolutionLayer(new MatrixDimension(8,8), templateSize, activation),
//                new ConvolutionLayer(new MatrixDimension(6,6), templateSize, activation),
//                new ConvolutionLayer(new MatrixDimension(4,4), templateSize, activation),
//                new ConvolutionLayer(new MatrixDimension(2,2), new MatrixDimension(2,2), activation)
//        );
//
//        LinearGraph errorsGraph = new LinearGraph("Total error");
//
//        int iteration = 0;
//        double error;
//        while ((error = net.backpropagate(inputs, outputs, inputs, outputs)) > 1e-2) {
//            if (iteration++ % 1000 == 0) {
//                errorsGraph.add(error);
//                System.out.println("Iteration " + iteration + " has error: " + error);
//            }
//        }
//
//        new Window(new VerticalContainer(new Row(errorsGraph)), true, true);
//    }
//}
