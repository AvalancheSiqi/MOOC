/*-----------------------------------------------------------------------------
 * Author:       Siqi Wu
 * Created:      14/02/2015
 * Last updated: 14/02/2015
 *
 * Compilation:  javac-algs4 Brute.java
 * Execution:    java-algs4 Brute < input.txt
 *
 * Examines 4 points at a time and checks whether they all lie on the same
 * line segment, printing out any such line segments to standard output and
 * drawing them using standard drawing.
 * 
 * Princeton University Algorithms Part I on Coursera
 *---------------------------------------------------------------------------*/

import java.util.Arrays;

public class Brute {
    private static boolean check(Point[] points) {
        return points[0].slopeTo(points[1]) == points[0].slopeTo(points[2])
            && points[0].slopeTo(points[1]) == points[0].slopeTo(points[3]);
    }
    
    private static void display(Point[] points) {
        int n = points.length;
        StdOut.print(points[0].toString());
        for (int i = 1; i < n; i++) {
            StdOut.print(" -> " + points[i].toString());
        }
        StdOut.println();
        points[0].drawTo(points[n-1]);
    }
    
    public static void main(String[] args) {
        // rescale coordinates and turn on animation mode
        StdDraw.setXscale(0, 32768);
        StdDraw.setYscale(0, 32768);
        StdDraw.show(0);
        StdDraw.setPenRadius(0.01);  // make the points a bit larger
        
        // read in the input
        String filename = args[0];
        In in = new In(filename);
        int n = in.readInt();
        Point[] points = new Point[n];
        for (int i = 0; i < n; i++) {
            int x = in.readInt();
            int y = in.readInt();
            Point p = new Point(x, y);
            p.draw();
            points[i] = p;
        }
        
        for (int i = 0; i < n-3; i++) {
            for (int j = i+1; j < n-2; j++) {
                for (int k = j+1; k < n-1; k++) {
                    for (int h = k+1; h < n; h++) {
                        Point[] temp = { points[i], points[j], points[k], points[h] };
                        if (check(temp)) {
                            Arrays.sort(temp);
                            display(temp);
                        }
                    }
                }
            }
        }
        
        // display to screen all at once
        StdDraw.show(0);

        // reset the pen radius
        StdDraw.setPenRadius();
    }
}