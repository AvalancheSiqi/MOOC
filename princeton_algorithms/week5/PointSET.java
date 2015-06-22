/*-----------------------------------------------------------------------------
 * Author:       Siqi Wu
 * Created:      06/03/2015
 * Last updated: 08/03/2015
 *
 * Compilation:  javac-algs4 PointSET.java
 * Execution:    java-algs4 PointSET < input.txt
 *
 * Represents a set of points in the unit square.
 * 
 * Princeton University Algorithms Part I on Coursera
 *---------------------------------------------------------------------------*/

import java.util.TreeSet;
import java.util.Iterator;

public class PointSET {
    private TreeSet<Point2D> rbTree;
    
    // construct an empty set of points
    public PointSET() {
        this.rbTree = new TreeSet<Point2D>();
    }
    
    // is the set empty?
    public boolean isEmpty() {
        return rbTree.isEmpty();
    }
    
    // number of points in the set 
    public int size() {
        return rbTree.size();
    }
    
    // add the point to the set (if it is not already in the set)
    public void insert(Point2D p) {
        if (p == null) throw new java.lang.NullPointerException();
        rbTree.add(p);
    }
    
    // does the set contain point p? 
    public boolean contains(Point2D p) {
        if (p == null) throw new java.lang.NullPointerException();
        return rbTree.contains(p);
    }
    
    // draw all points to standard draw 
    public void draw() {
        Iterator<Point2D> iterator = rbTree.iterator();
        StdDraw.clear();
        StdDraw.setPenColor(StdDraw.BLACK);
        StdDraw.setPenRadius(.01);
        while (iterator.hasNext()) {
            Point2D p = iterator.next();
            p.draw();
        }
    }
    
    // all points that are inside the rectangle 
    public Iterable<Point2D> range(RectHV rect) {
        if (rect == null) throw new java.lang.NullPointerException();
        TreeSet<Point2D> s = new TreeSet<Point2D>();
        Iterator<Point2D> iterator = rbTree.iterator();
        while (iterator.hasNext()) {
            Point2D p = iterator.next();
            if (rect.contains(p)) s.add(p);
        }
        return s;
    }
    
    // a nearest neighbor in the set to point p; null if the set is empty
    public Point2D nearest(Point2D p) {
        if (p == null) throw new java.lang.NullPointerException();
        if (this.isEmpty()) return null;
        Iterator<Point2D> iterator = rbTree.iterator();
        Point2D q = iterator.next();
        double dist = p.distanceSquaredTo(q);
        while (iterator.hasNext()) {
            Point2D k = iterator.next();
            if (dist > p.distanceSquaredTo(k)) {
                dist = p.distanceSquaredTo(k);
                q = k;
            }
        }
        return q;
    }

    // unit testing of the methods (optional) 
    public static void main(String[] args) {
        
        String filename = args[0];
        In in = new In(filename);
        
        PointSET brute = new PointSET();
        while (!in.isEmpty()) {
            double x = in.readDouble();
            double y = in.readDouble();
            Point2D p = new Point2D(x, y);
            brute.insert(p);
        }
        
        StdDraw.setPenColor(StdDraw.BLACK);
        StdDraw.setPenRadius(.01);
        brute.draw();
    }
}