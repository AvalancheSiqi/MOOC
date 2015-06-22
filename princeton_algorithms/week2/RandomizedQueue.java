/*-----------------------------------------------------------------------------
 * Author:       Siqi Wu
 * Created:      11/02/2015
 * Last updated: 12/02/2015
 * 
 * Compliation:  javac-algs4 RandomizedQueue.java
 * Execution:    java-algs4 RandomizedQueue < input.txt
 * 
 * A program implements programming assignment wee2, adding and removing item
 * removed chosen uniformly at random from items in the data structure.
 * 
 * Princeton University Algorithms Part I on Coursera
 *---------------------------------------------------------------------------*/

import java.util.Iterator;
import java.util.NoSuchElementException;

public class RandomizedQueue<Item> implements Iterable<Item> {
    private int n;          // size of the randomized queue
    private Item[] a;     // number of elements on randomized queue
    
    // construct an empty randomized queue
    public RandomizedQueue() {
        n = 0;
        a = (Item[]) new Object[2];
        assert check();
    }
    
    // is the queue empty?
    public boolean isEmpty() { return this.n == 0; }
    
    // return the number of items on the queue
    public int size() { return this.n; }
    
    // resize the underlying array holding the elements
    private void resize(int capacity) {
        assert capacity >= n;
        Item[] temp = (Item[]) new Object[capacity];
        for (int i = 0; i < n; i++) {
            temp[i] = a[i];
        }
        a = temp;
    }
    
    // add the item
    public void enqueue(Item item) {
        if (item == null) throw new NullPointerException();
        if (n == a.length) resize(2*a.length);
        a[n++] = item;
        assert check();
    }
    
    // remove and return a random item
    public Item dequeue() {
        if (isEmpty()) throw new NoSuchElementException();
        int r = StdRandom.uniform(this.n);
        Item item = a[r];
        a[r] = a[n-1];
        a[n-1] = null; // to avoid loitering
        n--;
        // shrink size of array if necessary
        if (n > 0 && n == a.length/4) resize(a.length/2);
        assert check();
        return item;
    }
    
    // return (but do not remove) a random item
    public Item sample() {
        if (n == 0) throw new NoSuchElementException();
        int r = StdRandom.uniform(this.n);
        return a[r];
    }
    
    // return an independent iterator over items in random order
    public Iterator<Item> iterator() { return new RandomizedQueueIterator(); }

    private class RandomizedQueueIterator implements Iterator<Item> {
        private int i;
        private int[] order;
        
        public RandomizedQueueIterator() {
            i = n;
            order = new int[n];
            for (int j = 0; j < n; j++) {
                order[j] = j;
            }
            StdRandom.shuffle(order);
        }

        public boolean hasNext() { return i > 0; }

        public void remove() { throw new UnsupportedOperationException(); }

        public Item next() {
            if (!hasNext()) throw new NoSuchElementException();
            return a[order[--i]];
        }
    }
    
    // check internal invariants
    private boolean check() {
        for (int i = 0; i < a.length; i++) {
            if (i < n && a[i] == null) return false;
            if (i >= n && a[i] != null) return false;
        }
        return true;
    }
    
    // unit testing
    public static void main(String[] args) {
        RandomizedQueue<String> rq = new RandomizedQueue<String>();
        while (!StdIn.isEmpty()) {
            String item = StdIn.readString();
            if (!item.equals("-")) rq.enqueue(item);
            else if (!rq.isEmpty()) StdOut.println(rq.dequeue() + " ");
        }
        StdOut.println("(" + rq.size() + " left on randomized queue)");
        StdOut.print("iterator: ");
        Iterator iter = rq.iterator();
        while (iter.hasNext()) {
            StdOut.print(iter.next() + " ");
        }
        StdOut.println();
    }
}