/*-----------------------------------------------------------------------------
 * Author:       Siqi Wu
 * Created:      11/02/2015
 * Last updated: 11/02/2015
 * 
 * Compliation:  javac-algs4 Deque.java
 * Execution:    java-algs4 Deque < input.txt
 * 
 * A program implements programming assignment wee2, a generalization of a
 * stack and a queue that supports adding and removing items from either the
 * front or the back of the data structure. 
 * 
 * Princeton University Algorithms Part I on Coursera
 *---------------------------------------------------------------------------*/

import java.util.Iterator;
import java.util.NoSuchElementException;

public class Deque<Item> implements Iterable<Item> {
    private int n;          // size of the deque
    private Node first;     // front of the deque
    private Node last;      // back of the deque
    
    // helper node class
    private class Node {
        private Item item;
        private Node ahead;
        private Node next;
    }
    
    // construct an empty deque
    public Deque() {
        n = 0;
        first = null;
        last = null;
        assert check();
    }
    
    // is the deque empty?
    public boolean isEmpty() { return this.n == 0; }
    
    // return the number of items on the deque
    public int size() { return this.n; }
    
    // add the item to the front
    public void addFirst(Item item) {
        if (item == null) throw new NullPointerException();
        Node newNode = new Node();
        newNode.item = item;
        if (n == 0) {
            first = newNode;
            last = newNode;
        }
        else {
            Node oldFirst = first;
            oldFirst.ahead = newNode;
            newNode.next = oldFirst;
            first = newNode;
        }
        n++;
        assert check();
    }
    
    // add the item to the end
    public void addLast(Item item) {
        if (item == null) throw new NullPointerException();
        Node newNode = new Node();
        newNode.item = item;
        if (n == 0) {
            first = newNode;
            last = newNode;
        }
        else {
            Node oldLast = last;
            oldLast.next = newNode;
            newNode.ahead = oldLast;
            last = newNode;
        }
        n++;
        assert check();
    }
    
    // remove and return the item from the front
    public Item removeFirst() {
        if (n == 0) throw new NoSuchElementException();
        Item item = first.item;
        if (n == 1) {
            first = null;
            last = null;
        }
        else {
            first = first.next;
            first.ahead = null;
        }
        n--;
        assert check();
        return item;
    }
    
    // remove and return the item from the end
    public Item removeLast() {
        if (n == 0) throw new NoSuchElementException();
        Item item = last.item;
        if (n == 1) {
            first = null;
            last = null;
        }
        else {
            last = last.ahead;
            last.next = null;
        }
        n--;
        assert check();
        return item;
    }
    
    // return an iterator over items in order from front to end    
    public Iterator<Item> iterator() { return new DequeIterator(); }

    private class DequeIterator implements Iterator<Item> {
        private Node current = first;
        public boolean hasNext() { return current != null; }
        public void remove() { throw new UnsupportedOperationException(); }
        public Item next() {
            if (!hasNext()) throw new NoSuchElementException();
            Item item = current.item;
            current = current.next; 
            return item;
        }
    }
    
    // check internal invariants
    private boolean check() {
        if (n == 0) {
            if (first != null | last != null) return false;
        }
        else if (n == 1) {
            if (first == null | last == null) return false;
            if (first.ahead != null | first.next != null | last.ahead != null
                    | last.next != null) return false;
        }
        else {
            if (first.next == null || last.ahead == null) return false;
        }

        // check internal consistency of instance variable N
        int numberOfNodes = 0;
        for (Node x = first; x != null; x = x.next) {
            numberOfNodes++;
        }
        if (numberOfNodes != n) return false;
        
        return true;
    }
    
    // unit testing
    public static void main(String[] args) {
        Deque<String> d = new Deque<String>();
        while (!StdIn.isEmpty()) {
            String item = StdIn.readString();
            if (!item.equals("-")) d.addLast(item);
            else if (!d.isEmpty()) StdOut.println(d.removeLast() + " ");
        }
        StdOut.println("(" + d.size() + " left on deque)");
        StdOut.print("iterator: ");
        Iterator iter = d.iterator();
        while (iter.hasNext()) {
            StdOut.print(iter.next() + " ");
        }
        StdOut.println();
    }
}