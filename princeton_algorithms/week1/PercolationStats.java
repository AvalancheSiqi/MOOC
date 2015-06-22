/*-----------------------------------------------------------------------------
 * Author:       Siqi Wu
 * Created:      04/02/2015
 * Last updated: 04/02/2015
 * 
 * Compliation:  javac-algs4 PercolationStats.java
 * Execution:    java-algs4 PercolationStats N T
 * 
 * Performs T independent computational experiments on an N-by-N grid
 * 
 * Princeton University Algorithm Part I on Coursera
 *---------------------------------------------------------------------------*/

public class PercolationStats {
    private double[] record;
    private int t;
    
    // perform T independent experiments on an N-by-N grid
    public PercolationStats(int N, int T) {
        if (N <= 0 || T <= 0) throw new IllegalArgumentException("illegal arguments");
        this.t = T;
        record = new double[T];
        for (int k = 0; k < T; k++) {
            Percolation percolation = new Percolation(N);
            int cnt = 0;
            while (!percolation.percolates()) {
                int row = (int) (Math.random() * N) + 1;
                int col = (int) (Math.random() * N) + 1;
                if (!percolation.isOpen(row, col)) {
                    percolation.open(row, col);
                    cnt++;
                }
            }
            record[k] = 1.0*cnt/(N*N);
        }
    }
    
    // sample mean of percolation threshold
    public double mean() {
        double sum = 0.0;
        for (double k : this.record) {
            sum += k;
        }
        return sum/this.t;
    }
    
    // sample standard deviation of percolation threshold
    public double stddev() {
        double u = this.mean();
        double sum = 0.0;
        for (double k : this.record) {
            sum += (k - u)*(k - u);
        }
        return Math.sqrt(sum/(this.t - 1));
    }
    
    // low endpoint of 95% confidence interval
    public double confidenceLo() {
        double u = this.mean();
        double alpha = this.stddev();
        return u - 1.96*alpha/Math.sqrt(this.t);
    }
    
    // high endpoint of 95% confidence interval
    public double confidenceHi() {
        double u = this.mean();
        double alpha = this.stddev();
        return u + 1.96*alpha/Math.sqrt(this.t);
    }
    
    public static void main(String[] args) {
        int N = Integer.parseInt(args[0]);
        int T = Integer.parseInt(args[1]);
        PercolationStats p = new PercolationStats(N, T);
        
        System.out.println("mean                    = " + p.mean());
        System.out.println("stddev                  = " + p.stddev());
        System.out.println("95% confidence interval = " + p.confidenceLo() + ", " + p.confidenceHi());
    }
}