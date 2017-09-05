package com.lirfu.networks.datatransformations;

import com.lirfu.graphicslib.matrix.IMatrix;
import com.lirfu.graphicslib.matrix.Matrix;

import java.io.*;

/**
 * Created by lirfu on 05.09.17..
 */
public class BinaryImage {
    private boolean[][] binaryArray;

    public BinaryImage(File file) throws MalformedFileException {
        try {

            InputStreamReader reader = new InputStreamReader(new FileInputStream(file));
            int input;

            int rows = 0, cols = 0;
            boolean colsFinished = false;
            while ((input = reader.read()) != '\n') {
                if (input == ' ') {
                    colsFinished = true;
                } else if (colsFinished) {
                    rows *= 10;
                    rows += input - '0';
                } else {
                    cols *= 10;
                    cols += input - '0';
                }
            }

            binaryArray = new boolean[rows][cols];
            System.out.println("Image " + file.getName() + " size: " + cols + "x" + rows);

            int line = 0, index = 0;
            while ((input = reader.read()) != -1) {
                if (input == '0') {
                    binaryArray[line][index++] = false;
                } else if (input == '1') {
                    binaryArray[line][index++] = true;
                } else if (input == '\n') {
                    index = 0;
                    line++;
                }
            }

        } catch (IOException e) {
            throw new MalformedFileException(e.getMessage());
        }
    }

    public IMatrix toMatrix() {
        Matrix matrix = new Matrix(binaryArray.length, binaryArray[0].length);

        for (int r = 0; r < matrix.getRowsCount(); r++)
            for (int c = 0; c < matrix.getColsCount(); c++)
                matrix.set(r, c, binaryArray[r][c] ? 1 : 0);

        return matrix;
    }
}
