/*-----------------------------------------------------------------------------
 * Author:       Siqi Wu
 * Created:      11/02/2015
 * Last updated: 12/02/2015
 * 
 * Compliation:  javac-algs4 Subset.java
 * Execution:    java-algs4 Subset < integer
 * 
 * A program implements programming assignment wee2, takes a command-line
 * integer k; reads in a sequence of N strings from standard input using
 * StdIn.readString(); and prints out exactly k of them, uniformly at random.
 * Using the idea of reservoir sampling.
 * 
 * Princeton University Algorithms Part I on Coursera
 *---------------------------------------------------------------------------*/

public class Subset {
    public static void main(String[] args) {
        int k = Integer.parseInt(args[0]);
        if (k != 0) {
            int cnt = 0;
            RandomizedQueue<String> rq = new RandomizedQueue<String>();
            while (cnt < k && !StdIn.isEmpty()) {
                cnt++;
                String item = StdIn.readString();
                rq.enqueue(item);
            }
            while(!StdIn.isEmpty()) {
                String item = StdIn.readString();
                cnt++;
                int r = StdRandom.uniform(cnt) + 1;
                if (r <= k) {
                    rq.dequeue();
                    rq.enqueue(item);
                }
            }
            while (!rq.isEmpty()) {
                StdOut.println(rq.dequeue());
            }
        }
    }
}