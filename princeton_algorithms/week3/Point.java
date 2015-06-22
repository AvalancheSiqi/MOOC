/*-----------------------------------------------------------------------------
 * Author:       Siqi Wu
 * Created:      13/02/2015
 * Last updated: 14/02/2015
 *
 * Compilation:  javac-algs4 Point.java
 * Execution:    java-algs4 Point
 *
 * An immutable data type for points in the plane.
 * 
 * Princeton University Algorithms Part I on Coursera
 *---------------------------------------------------------------------------*/

import java.util.Comparator;

public class Point implements Comparable<Point> {
    // compare points by slope
    public final Comparator<Point> SLOPE_ORDER = new SlopeOrder();
    private final int x; // x coordinate
    private final int y; // y coordinate
    
    private class SlopeOrder implements Comparator<Point> {
        @Override
        public int compare(Point v, Point w) {
            if (slopeTo(v) < slopeTo(w)) return -1;
            else if (slopeTo(v) > slopeTo(w)) return +1;
            else return 0;
        }
    }
    
    // create the point (x, y)
    public Point(int x, int y) {
        this.x = x;
        this.y = y;
    }

    // plot this point to standard drawing
    public void draw() {
        StdDraw.point(x, y);
    }

    // draw line between this point and that point to standard drawing
    public void drawTo(Point that) {
        StdDraw.line(this.x, this.y, that.x, that.y);
    }

    // slope between this point and that point
    public double slopeTo(Point that) {
        if (this.y == that.y && this.x == that.x) return Double.NEGATIVE_INFINITY;
        else if (this.y == that.y) return (0.0);
        else if (this.x == that.x) return Double.POSITIVE_INFINITY;
        return 1.0 * (that.y - this.y) / (that.x - this.x);
    }

    // is this point lexicographically smaller than that one?
    // comparing y-coordinates and breaking ties by x-coordinates
    public int compareTo(Point that) {
        if (this.y < that.y || (this.y == that.y && this.x < that.x)) return -1;
        else if (this.y == that.y && this.x == that.x) return 0;
        else return +1;
    }

    // return string representation of this point
    public String toString() {
        return "(" + x + ", " + y + ")";
    }

    // unit test
    public static void main(String[] args) { }
}