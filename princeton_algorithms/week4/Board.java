/*-----------------------------------------------------------------------------
 * Author:       Siqi Wu
 * Created:      18/02/2015
 * Last updated: 20/02/2015
 *
 * Compilation:  javac-algs4 Board.java
 * Execution:    java-algs4 Board < input.txt
 *
 * Creating an immutable data type Board to support necessary ops.
 * 
 * Princeton University Algorithms Part I on Coursera
 *---------------------------------------------------------------------------*/

import java.util.LinkedList;
import java.util.Queue;

public class Board {
    private int[][] blocks;
    private int n;
    
    // construct a board from an N-by-N array of blocks
    // (where blocks[i][j] = block in row i, column j)
    public Board(int[][] blocks) {
        this.n = blocks.length;
        this.blocks = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                this.blocks[i][j] = blocks[i][j];
            }
        }
    }
    
    // board dimension N
    public int dimension() {
        return blocks.length;
    }
    
    // number of blocks out of place
    public int hamming() {
        int ham = 0;
        for (int i = 0; i < this.n; i++) {
            for (int j = 0; j < this.n; j++) {
                if (blocks[i][j] != i*n+j+1) {
                    ham++;
                }
            }
        }
        // remove the last item, meant to be different
        return --ham;
    }
    
    // sum of Manhattan distances between blocks and goal
    public int manhattan() {
        int man = 0;
        for (int i = 0; i < this.n; i++) {
            for (int j = 0; j < this.n; j++) {
                if (blocks[i][j] != 0) {
                    man += Math.abs(((blocks[i][j]-1) / n) - i);
                    man += Math.abs(((blocks[i][j]-1) % n) - j);
                }
            }
        }
        return man;
    }
    
    // is this board the goal board?
    public boolean isGoal() {
        if (blocks[n-1][n-1] != 0) return false;
        for (int i = 0; i < n-1; i++) {
            for (int j = 0; j < n; j++) {
                if (blocks[i][j] != i*n+j+1) return false;
            }
        }
        for (int j = 0; j < n-1; j++) {
            if (blocks[n-1][j] != n*n-n+j+1) return false;
        }
        return true;
    }
    
    // a boadr that is obtained by exchanging two adjacent blocks in the same row
    public Board twin() {
        int[][] twinBlocks = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                twinBlocks[i][j] = this.blocks[i][j];
            }
        }
        if (this.blocks[0][0] != 0 && this.blocks[0][1] != 0) {
            twinBlocks[0][0] = this.blocks[0][1];
            twinBlocks[0][1] = this.blocks[0][0];
        }
        else {
            twinBlocks[1][0] = this.blocks[1][1];
            twinBlocks[1][1] = this.blocks[1][0];
        }
        return new Board(twinBlocks);
    }
    
    // does this board equal y?
    public boolean equals(Object y) {
        if (y == this) return true;
        if (y == null) return false;
        if (y.getClass() != this.getClass()) return false;
        Board that = (Board) y;
        if (that.dimension() != this.dimension()) return false;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (that.blocks[i][j] != this.blocks[i][j]) return false;
            }
        }
        return true;
    }
    
    // all neighboring boards
    public Iterable<Board> neighbors() { 
        Queue<Board> queue = new LinkedList<Board>();
        int t = -1;
//        (t/2-1)^(t+1) = (-1)^1, (-1)^2, 0^3, 0^4 = -1, 1, 0, 0;
//        (-t/2)^(t+1) = 0^1, 0^2, -1^3, -1^4 = 0, 0, -1, 1;
        while (++t < 4) {
            int[][] neighborBlocks = new int[n][n];
            int blankX = 0;
            int blankY = 0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    neighborBlocks[i][j] = this.blocks[i][j];
                    if (this.blocks[i][j] == 0) {
                        blankX = i;
                        blankY = j;
                    }
                }
            }
            
            int newBlankX = blankX + (int) Math.pow(t/2-1, t+1);
            int newBlankY = blankY + (int) Math.pow(-t/2, t+1);
            if (newBlankX < 0 || newBlankX > n-1 || newBlankY < 0 || newBlankY > n-1) continue;
            neighborBlocks[blankX][blankY] = this.blocks[newBlankX][newBlankY];
            neighborBlocks[newBlankX][newBlankY] = this.blocks[blankX][blankY];
            Board neighbor = new Board(neighborBlocks);
            queue.add(neighbor);
        }
        return queue;
    }
    
    // string representation of this board (in the output format specified below)
    public String toString() {
        StringBuilder s = new StringBuilder();
        s.append(n + "\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                s.append(String.format("%2d ", this.blocks[i][j]));
            }
            s.append("\n");
        }
        return s.toString();
    }
    
    // unit tests
    public static void main(String[] args) {
        In in = new In(args[0]);
        int n = in.readInt();
        int[][] blocks = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                blocks[i][j] = in.readInt();
            }
        }
        Board initial = new Board(blocks);
        
        StdOut.println("dimension: " + initial.dimension());
        StdOut.println("hamming distance: " + initial.hamming());
        StdOut.println("manhattan distance: " + initial.manhattan());
        StdOut.println("isGoal: " + initial.isGoal());
        StdOut.println("toString: ");
        StdOut.print(initial.toString());
        StdOut.println("twin toString: ");
        StdOut.print(initial.twin().toString());
        StdOut.println("equals: " + initial.equals(initial.twin()));
        StdOut.println("neighbors: ");
        for (Board neighbor : initial.neighbors()) {
            StdOut.print(neighbor.toString());
        }
    }
}