/*-----------------------------------------------------------------------------
 * Author:       Siqi Wu
 * Created:      02/02/2015
 * Last updated: 05/02/2015
 * 
 * Compliation:  javac-algs4 Percolation.java
 * Execution:    java-algs4 Percolation $input
 * 
 * A program implements programming assignment week1, with the provided
 * WeightedQuickUnionUF API.
 * 
 * Princeton University Algorithm Part I on Coursera
 *---------------------------------------------------------------------------*/
public class Percolation {
    private int length;
    private WeightedQuickUnionUF canvas;
    private boolean[] openState;
    private boolean[] connectToTop;
    private boolean[] connectToBottom;
    private boolean isPercolated;
    
    // create N-by-N grid, with all sites blocked
    public Percolation(int length) {
        if (length <= 0) throw new IllegalArgumentException("illegal arguments");
        this.length = length;
        this.canvas = new WeightedQuickUnionUF(length*length);
        this.openState = new boolean[length*length];
        this.connectToTop = new boolean[length*length];
        this.connectToBottom = new boolean[length*length];
        this.isPercolated = false;
    }
    
    private boolean validate(int i, int j) {
        if (i > 0 && i <= length && j > 0 && j <= length) return true;
        return false;
    }
    
    private int convert(int i, int j) {
        return this.length*(i-1) + j-1;
    }
    
    private void link(int dst, int k, int h) {
        if (validate(k, h) && isOpen(k, h)) {
            int src = convert(k, h);
            int rootDst = canvas.find(dst);
            int rootSrc = canvas.find(src);
            canvas.union(dst, src);
            if (connectToTop[rootDst] || connectToTop[rootSrc]) {
                connectToTop[rootDst] = true;
                connectToTop[rootSrc] = true;
            }
            if (connectToBottom[rootDst] || connectToBottom[rootSrc]) {
                connectToBottom[rootDst] = true;
                connectToBottom[rootSrc] = true;
            }
        }
    }
    
    // open site (row i, column j) if it is not open already
    public void open(int i, int j) {
        if (validate(i, j)) {
            int point = convert(i, j);
            openState[point] = true;
            
            // if open top
            if (i == 1) connectToTop[point] = true;
            // if open bottom
            if (i == length) connectToBottom[point] = true;
            
            link(point, i-1, j);
            link(point, i+1, j);
            link(point, i, j-1);
            link(point, i, j+1);
            
            int root = canvas.find(point);
            if (connectToTop[root] && connectToBottom[root]) isPercolated = true;
        }
        else {
            throw new IndexOutOfBoundsException("open: index i, j out of bounds");
        }
    }
    
    // is site (row i, column j) open?
    public boolean isOpen(int i, int j) {
        if (validate(i, j)) return openState[convert(i, j)];
        else {
            throw new IndexOutOfBoundsException("isOpen: index i, j out of bounds");
        }
    }
    
    // is site (row i, column j) full?
    public boolean isFull(int i, int j) {
        if (validate(i, j)) return connectToTop[canvas.find(convert(i, j))];
        else {
            throw new IndexOutOfBoundsException("isFull: index i, j out of bounds");
        }
    }

    // does the system percolate?
    public boolean percolates() {
        return isPercolated;
    }
    
//    public static void main(String[] args) {
//        int n = 8;
//        Percolation percolation = new Percolation(n);
//        percolation.open(2, 1);
//        System.out.println(percolation.isFull(2, 1));
//        percolation.open(1, 1);
//        System.out.println(percolation.isFull(1, 1));
//        int cnt = 0;
//        while (!percolation.percolates()) {
//                int row = (int) (Math.random() * n) + 1;
//                int col = (int) (Math.random() * n) + 1;
//                if (!percolation.isOpen(row, col)) {
//                    percolation.open(row, col);
//                    cnt++;
//                }
//            }
//        System.out.println(cnt);
//        System.out.println(percolation.union);
//        System.out.println(percolation.cf);
//    }
}