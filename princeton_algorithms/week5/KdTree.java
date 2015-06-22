/*-----------------------------------------------------------------------------
 * Author:       Siqi Wu
 * Created:      06/03/2015
 * Last updated: 08/03/2015
 *
 * Compilation:  javac-algs4 KdTree.java
 * Execution:    java-algs4 KdTree < input.txt
 *
 * Represents a set of points in the unit square.
 * 
 * Princeton University Algorithms Part I on Coursera
 *---------------------------------------------------------------------------*/

import java.util.TreeSet;

public class KdTree {
    private Node root;
    private int size;
    private static final RectHV CONTAINER = new RectHV(0, 0, 1, 1);  
    
    private static class Node {
        private Point2D p;          // the point
        private Node lb;            // the left/bottom subtree
        private Node rt;            // the right/top subtree
        private boolean isVertical; // the orientation of kdtree root
        
        public Node(Point2D p, Node lb, Node rt, boolean isVertical) {
            this.p = p;
            this.lb = lb;
            this.rt = rt;
            this.isVertical = isVertical;
        }
    }
    
    // construct an empty set of points
    public KdTree() {
        this.root = null;
        this.size = 0;
    }
    
    // is the set empty?
    public boolean isEmpty() {
        return this.size == 0;
    }
    
    // number of points in the set 
    public int size() {
        return this.size;
    }
    
    // add the point to the set (if it is not already in the set)
    public void insert(Point2D p) {
        if (p == null) throw new java.lang.NullPointerException();
        this.root = insert(p, this.root, true);
    }
    
    private Node insert(Point2D p, Node n, boolean isVertical) {
        // if new node, create it 
        if (n == null) {
            size += 1;
            return new Node(p, null, null, isVertical);
        }
        
        // if already in, return it
        if (n.p.equals(p)) {
            return n;
        }
        
        // if at the left / bottom part
        if ((n.isVertical && Point2D.X_ORDER.compare(p, n.p) < 0)
                || (!n.isVertical && Point2D.Y_ORDER.compare(p, n.p) < 0)) {
            n.lb = insert(p, n.lb, !n.isVertical);
        } else {
            n.rt = insert(p, n.rt, !n.isVertical);
        }
        return n;
    }
    
    // does the set contain point p? 
    public boolean contains(Point2D p) {
        if (p == null) throw new java.lang.NullPointerException();
        return contains(p, root);
    }
    
    private boolean contains(Point2D p, Node n) {
        if (n == null) return false;
        
        if (p.equals(n.p)) return true;
        
        if ((n.isVertical && Point2D.X_ORDER.compare(p, n.p) < 0)
                || (!n.isVertical && Point2D.Y_ORDER.compare(p, n.p) < 0)) {
            return contains(p, n.lb);
        } else {
            return contains(p, n.rt);
        }
    }
    
    // draw all points to standard draw 
    public void draw() {
        StdDraw.setPenColor(StdDraw.BLACK);  
        StdDraw.setPenRadius();  
        CONTAINER.draw();
        
        this.draw(root, CONTAINER);
    }
    
    private void draw(Node n, RectHV rect) {
        if (n == null) return;
        
        StdDraw.setPenColor(StdDraw.BLACK);
        StdDraw.setPenRadius(.01);
        n.p.draw();
        
        // get the min and max points of division line 
        Point2D min, max;
        if (n.isVertical) {  
            StdDraw.setPenColor(StdDraw.RED);  
            min = new Point2D(n.p.x(), rect.ymin());  
            max = new Point2D(n.p.x(), rect.ymax());  
        } else {
            StdDraw.setPenColor(StdDraw.BLUE);  
            min = new Point2D(rect.xmin(), n.p.y());  
            max = new Point2D(rect.xmax(), n.p.y());  
        } 
        
        // draw that division line  
        StdDraw.setPenRadius();  
        min.drawTo(max);
        
        // recursively draw two parts
        draw(n.lb, leftRect(rect, n));
        draw(n.rt, rightRect(rect, n));
    }
    
    private RectHV leftRect(RectHV rect, Node n) {
        if (n.isVertical) {  
            return new RectHV(rect.xmin(), rect.ymin(), n.p.x(), rect.ymax());  
        } else {  
            return new RectHV(rect.xmin(), rect.ymin(), rect.xmax(), n.p.y());  
        } 
    }
    
    private RectHV rightRect(RectHV rect, Node n) {
        if (n.isVertical) {  
            return new RectHV(n.p.x(), rect.ymin(), rect.xmax(), rect.ymax());  
        } else {  
            return new RectHV(rect.xmin(), n.p.y(), rect.xmax(), rect.ymax());  
        } 
    }
    
    // all points that are inside the rectangle 
    public Iterable<Point2D> range(RectHV rect) {
        if (rect == null) throw new java.lang.NullPointerException();
        TreeSet<Point2D> rangeSet = new TreeSet<Point2D>();
        range(root, CONTAINER, rect, rangeSet);
        return rangeSet;
    }
    
    private void range(Node n, RectHV container, RectHV rect, TreeSet<Point2D> rangeSet) {
        if (n == null) return;
        
        if (rect.intersects(container)) {
            if (rect.contains(n.p)) rangeSet.add(n.p);
            range(n.lb, leftRect(container, n), rect, rangeSet);
            range(n.rt, rightRect(container, n), rect, rangeSet);
        }
    }
    
    // a nearest neighbor in the set to point p; null if the set is empty
    public Point2D nearest(Point2D p) {
        if (p == null) throw new java.lang.NullPointerException();
        if (this.isEmpty()) return null;
        return nearest(root, CONTAINER, p, null);
    }
    
    private Point2D nearest(Node n, RectHV rect, Point2D p, Point2D candidate) {
        if (n == null) return candidate;
        
        double distOfQueryNearest = 0.0;
        double distOfQueryRect = 0.0;
        RectHV left = null;  
        RectHV right = null;
        Point2D nearest = candidate;
        
        if (nearest != null) {  
            distOfQueryNearest = p.distanceSquaredTo(nearest);  
            distOfQueryRect = rect.distanceSquaredTo(p);  
        }
        
        if (nearest == null || distOfQueryNearest > distOfQueryRect) {
            if (nearest == null) {
                nearest = n.p;
                return nearest;
            }
            if (distOfQueryNearest > p.distanceSquaredTo(n.p)) nearest = n.p;
            
            left = leftRect(rect, n);
            right = rightRect(rect, n); 
            if (n.isVertical) {
                if (Point2D.X_ORDER.compare(p, n.p) < 0) {  
                    nearest = nearest(n.lb, left, p, nearest);  
                    nearest = nearest(n.rt, right, p, nearest);  
                } else {  
                    nearest = nearest(n.lb, right, p, nearest);  
                    nearest = nearest(n.rt, left, p, nearest);  
                }
            } else {
                if (Point2D.Y_ORDER.compare(p, n.p) < 0) {  
                    nearest = nearest(n.lb, left, p, nearest);  
                    nearest = nearest(n.rt, right, p, nearest);  
                } else {  
                    nearest = nearest(n.rt, right, p, nearest);  
                    nearest = nearest(n.lb, left, p, nearest);  
                }
            }
        }
        
        return nearest;
    }

    // unit testing of the methods (optional) 
    public static void main(String[] args) {
        
        String filename = args[0];
        In in = new In(filename);
        
        KdTree kdtree = new KdTree();
        while (!in.isEmpty()) {
            double x = in.readDouble();
            double y = in.readDouble();
            Point2D p = new Point2D(x, y);
            kdtree.insert(p);
            StdOut.println("contains: " + kdtree.contains(p));
            StdOut.println("size: " + kdtree.size());
            StdOut.println("---------------------------");
        }
        
        StdDraw.setPenColor(StdDraw.BLACK);
        StdDraw.setPenRadius(.01);
        kdtree.draw();
    }
}