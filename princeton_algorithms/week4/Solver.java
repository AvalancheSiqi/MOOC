/*-----------------------------------------------------------------------------
 * Author:       Siqi Wu
 * Created:      18/02/2015
 * Last updated: 20/02/2015
 *
 * Compilation:  javac-algs4 Solver.java
 * Execution:    java-algs4 Solver < input.txt
 *
 * Checks whether points all lie on the same line segment, printing out any
 * such line segments to standard output and drawing them using standard
 * drawing. A faster implementation.
 * 
 * Princeton University Algorithms Part I on Coursera
 *---------------------------------------------------------------------------*/

import java.util.Stack;
import java.util.LinkedList;
import java.util.Queue;

public class Solver {
    private MinPQ<SearchNode> pq;
    private MinPQ<SearchNode> pqTwin;
    private Stack<SearchNode> explored;
    private Stack<SearchNode> exploredTwin;
    private Stack<Board> shortestPathStack;
    private Queue<Board> shortestPathQueue;
    private boolean solvability;
    private int minMoves;
    
    private class SearchNode implements Comparable<SearchNode> {
        private Board board;
        private int moves;
        private int priority;
        private Board prevBoard;
        
        public SearchNode(Board board, int moves, Board prevBoard) {
            this.board = board;
            this.moves = moves;
            this.priority = board.manhattan() + moves;
            this.prevBoard = prevBoard;
        }
        
        public int compareTo(SearchNode that) {
            if (this.priority < that.priority) return -1;
            else if (this.priority == that.priority) return 0;
            else return +1;
        }
    }
    
    // find a solution to the initial board (using the A* algorithm)
    public Solver(Board initial) {
        if (initial == null) throw new NullPointerException();
        this.solvability = false;
        this.minMoves = -1;
        this.explored = new Stack<SearchNode>();
        this.exploredTwin = new Stack<SearchNode>();
        this.shortestPathStack = new Stack<Board>();
        this.shortestPathQueue = new LinkedList<Board>();
        // initial search node priority queue
        this.pq = new MinPQ<SearchNode>();
        SearchNode start = new SearchNode(initial, 0, null);
        pq.insert(start);
        SearchNode sn = null;
        // initial twin search node priority queue
        this.pqTwin = new MinPQ<SearchNode>();
        SearchNode startTwin = new SearchNode(initial.twin(), 0, null);
        pqTwin.insert(startTwin);
        SearchNode snTwin = null;
        while ((sn = pq.delMin()).board.manhattan() != 0 && (snTwin = pqTwin.delMin()).board.manhattan() != 0) {
            explored.push(sn);
            exploredTwin.push(snTwin);
            Board currentBoard = sn.board;
            int currentMoves = sn.moves;
            Board prevBoard = sn.prevBoard;
            currentMoves++;
            for (Board successor : currentBoard.neighbors()) {
                if (!successor.equals(prevBoard)) {
                    pq.insert(new SearchNode(successor, currentMoves, currentBoard));
                }
            }
            Board currentBoardTwin = snTwin.board;
            int currentMovesTwin = snTwin.moves;
            Board prevBoardTwin = snTwin.prevBoard;
            currentMovesTwin++;
            for (Board successor : currentBoardTwin.neighbors()) {
                if (!successor.equals(prevBoardTwin)) {
                    pqTwin.insert(new SearchNode(successor, currentMovesTwin, currentBoardTwin));
                }
            }
        }
        if (sn.board.manhattan() == 0) {
            solvability = true;
            minMoves = sn.moves;
            explored.push(sn);
            shortestPathQueue = solutionHelper(explored);
        }
        else {
            solvability = false;
        }
    }
    
    private Queue<Board> solutionHelper(Stack<SearchNode> stack) {
        SearchNode node = stack.pop();
        if (node.moves == 0) return null;
        Board currentBoard = node.board;
        Board prevBoard = node.prevBoard;
        shortestPathStack.push(currentBoard);
        while (!stack.empty()) {
            node = stack.pop();
            if (prevBoard.equals(node.board)) {
                currentBoard = node.board;
                prevBoard = node.prevBoard;
                shortestPathStack.push(currentBoard);
            }
        }
        while (!shortestPathStack.empty()) {
            shortestPathQueue.offer(shortestPathStack.pop());
        }
        return shortestPathQueue;
    }
    
    // is the initial board solvable?
    public boolean isSolvable() {
        return solvability;
    }
    
    // min number of moves to solve initial board; -1 if unsolvable
    public int moves() {
        if (!solvability) return -1;
        return minMoves;
    }
    
    // sequence of boards in a shortest solution; null if unsolvable
    public Iterable<Board> solution() {
        if (!solvability) return null;
        return shortestPathQueue;
    }
    
    // solve a slider puzzle
    public static void main(String[] args) {
        // create initial board from file
        In in = new In(args[0]);
        int n = in.readInt();
        int[][] blocks = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                blocks[i][j] = in.readInt();
            }
        }
        Board initial = new Board(blocks);

        // solve the puzzle
        Solver solver = new Solver(initial);
        
        // print solution to standard output
        if (!solver.isSolvable())
            StdOut.println("No solution possible");
        else {
            StdOut.println("Minimum number of moves = " + solver.moves());
            for (Board board : solver.solution())
                StdOut.println(board);
        }
    }
}