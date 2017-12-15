package com.lirfu.networks.datatransformations;

import com.lirfu.lirfugraph.Point2D;

import java.awt.*;
import java.util.ArrayList;

public class Gesture {
    private ArrayList<Point2D> points_;

    public Gesture(ArrayList<Point> points) {
        points_ = new ArrayList<>();
        for (Point p : points)
            points_.add(new Point2D(p.x, p.y));
    }

    public ArrayList<Point2D> getPoints() {
        return points_;
    }

    public void center() {
        Point2D center = new Point2D(0, 0);
        for (Point2D p : points_)
            center.add(p);
        center.multiply(1. / points_.size());
        for (Point2D p : points_)
            p.sub(center);
    }

    public void scale() {
        double max = 0;
        for (Point2D p : points_) {
            if (max < Math.abs(p.x))
                max = Math.abs(p.x);
            if (max < Math.abs(p.y))
                max = Math.abs(p.y);
        }
        for (Point2D p : points_)
            p.multiply(1. / max);
    }

    public double getLength() {
        double length = 0;
        for (int i = 1; i < points_.size(); i++)
            length += points_.get(i - 1).distanceFrom(points_.get(i));
        return length;
    }

    public ArrayList<Point2D> selectRepresentative(int size) {
        ArrayList<Point2D> representative = new ArrayList<>();

        double delta = getLength() / (size - 1);

        for (int k = 0; k < size; k++) {
            double distance = 0;
            for (int i = 0; i < points_.size(); i++) {
                if (Math.abs(distance - k * delta) < 1e-2) { // equals
                    representative.add(new Point2D(points_.get(i)));
                    break;
                } else if (distance > k * delta && i > 0) { // somewhere in between
//                    Point2D interpolated = new Point2D(0, 0).add(points_.get(i)).sub(points_.get(i+1)).multiply(0.5);
//                    representative.add(interpolated);
                    representative.add(new Point2D(points_.get(i)));
                    break;
                }

                if (i == points_.size() - 1)
                    representative.add(points_.get(points_.size() - 1));
                distance += points_.get(i).distanceFrom(points_.get(i + 1));
            }
        }

        return representative;
    }

    public ArrayList<Point2D> getStandardizedRepresentative(int size) {
        center();
        scale();
        return selectRepresentative(size);
    }
}
