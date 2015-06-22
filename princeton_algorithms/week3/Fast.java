/*-----------------------------------------------------------------------------
 * Author:       Siqi Wu
 * Created:      14/02/2015
 * Last updated: 14/02/2015
 *
 * Compilation:  javac-algs4 Fast.java
 * Execution:    java-algs4 Fast < input.txt
 *
 * Checks whether points all lie on the same line segment, printing out any
 * such line segments to standard output and drawing them using standard
 * drawing. A faster implementation.
 * 
 * Princeton University Algorithms Part I on Coursera
 *---------------------------------------------------------------------------*/

import java.util.Arrays;

public class Fast {
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
        Point[] stds = new Point[n];
        for (int i = 0; i < n; i++) {
            int x = in.readInt();
            int y = in.readInt();
            Point p = new Point(x, y);
            Point s = new Point(x, y);
            p.draw();
            points[i] = p;
            stds[i] = s;
        }
        
        for (int i = 0; i < n; i++) {
            Arrays.sort(points, stds[i].SLOPE_ORDER);
            double[] values = new double[n];
            for (int j = 0; j < n; j++) {
                values[j] = stds[i].slopeTo(points[j]);
            }
            int k = 1;
            while (k < n) {
                boolean flag = true;
                int cnt = 0;
                for (int h = 1; h < n; h++) {
                    if (values[k] == values[h]) {
                        // take into account only when std is the first node
                        if (stds[i].compareTo(points[h]) < 0) {
                            cnt++;
                        }
                        else {
                            flag = false;
                            break;
                        }
                    }
                }
                if (cnt > 2 && flag) {
                    Point[] line = new Point[cnt+1];
                    int index = 0;
                    line[index++] = stds[i];
                    for (int h = 1; h < n; h++) {
                        if (values[k] == values[h]) {
                            line[index++] = points[h];
                        }
                    }
                    Arrays.sort(line);
                    StdOut.print(line[0].toString());
                    for (int m = 1; m < cnt+1; m++) {
                        StdOut.print(" -> " + line[m].toString());
                    }
                    StdOut.println();
                    line[0].drawTo(line[cnt]);
                }
                k += (cnt+1);
            }
        }
        
        // display to screen all at once
        StdDraw.show(0);

        // reset the pen radius
        StdDraw.setPenRadius();
    }
}