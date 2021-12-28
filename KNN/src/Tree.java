import java.io.*;
import java.util.*;

public class Tree {

    public static class Item {
        List<Integer> features;
        int cl;

        Item(List<Integer> features, int cl) {
            this.features = features;
            this.cl = cl;
        }
    }

    public static class DecisionTree {
        private class Node {
            boolean isLeaf;

            int ruleInd;
            double ruleValue;

            Node left, right;

            Node(int ruleInd, double ruleValue, Node left, Node right) {
                this.ruleInd = ruleInd;
                this.ruleValue = ruleValue;
                this.left = left;
                this.right = right;
                this.isLeaf = false;
            }

            Node(int cl) {
                this(cl, -1, null, null);
                this.isLeaf = true;
            }
        }

        private Node root;
        private int N, M, K;

        DecisionTree(int n, int m, int k) {
            M = m;
            N = n;
            K = k;
        }

        private int[] countClasses(List<Item> data) {
            int[] list = new int[K];
            for (Item item : data) {
                list[item.cl - 1]++;
            }
            return list;
        }

        private double qualitySet(int[] counts) {
            int sum = 0;
            for (int i : counts)
                sum += i;

            double ans = 1;
            for (int i : counts) {
                double p = (double)i / sum;
                ans -= p * p;
            }
            return ans;
        }

        private double quality(int S, int left, int right, int[] count, int[] countLeft, int[] countRight) {
            double ret = qualitySet(count);
            ret -= (double) left / S * qualitySet(countLeft);
            ret -= (double) right / S * qualitySet(countRight);

            return ret;
        }


        static final Random RND = new Random();
        private void swap(List<Item> array, int i, int j) {
            Item tmp = array.get(i);
            array.set(i, array.get(j));
            array.set(j, tmp);
        }

        private int partition(List<Item> array, int begin, int end, int f) {
            int index = begin + RND.nextInt(end - begin + 1);
            Item pivot = array.get(index);
            swap(array, index, end);
            for (int i = index = begin; i < end; ++ i) {
                if (array.get(i).features.get(f) <= pivot.features.get(f)) {
                    swap(array, index++, i);
                }
            }
            swap(array, index, end);
            return (index);
        }

        private void qsort(List<Item> array, int begin, int end, int f) {
            if (end > begin) {
                int index = partition(array, begin, end, f);
                qsort(array, begin, index - 1, f);
                qsort(array, index + 1,  end, f);
            }
        }

        private void mySort(List<Item> data, int f) {
            qsort(data, 0, data.size() - 1, f);
        }


        private Node trainImpl(List<Item> data, int h) {
            int counts[] = countClasses(data);
            if (h == 11) {
                int best = 0;
                int mxBest = counts[0];
                for (int i = 1; i < K; i++) {
                    if (counts[i] > mxBest) {
                        mxBest = counts[i];
                        best = i;
                    }
                }
                return new Node(best);
            }
            int cnt = 0;
            for (int i : counts) {
                if (i != 0) {
                    cnt++;
                }
            }
            if (cnt == 1) {
                for (int i = 0; i < K; i++) {
                    if (counts[i] != 0) {
                        return new Node(i);
                    }
                }
            }

            Double bestQlt = null;
            int bestRuleInd = -1;
            int bestRuleValue = -1;
            List<Item> bestLeft = new ArrayList<>();
            List<Item> bestRight = new ArrayList<>();

            for (int j = 0; j < M; j++) {
                mySort(data, j);

                int countsLeft[] = new int[K];
                int countsRight[] = counts.clone();
                Item cur = data.get(0);
                countsLeft[cur.cl - 1]++;
                countsRight[cur.cl - 1]--;

                for (int i = 1; i < data.size(); i++) {
                    double qlt = quality(data.size(), i, data.size() - i, counts, countsLeft, countsRight);

                    if (bestQlt == null || qlt > bestQlt) {
                        bestQlt = qlt;
                        bestLeft = new ArrayList<>(data.subList(0, i));
                        bestRight = new ArrayList<>(data.subList(i, data.size()));
                        bestRuleInd = j;
                        bestRuleValue = data.get(i - 1).features.get(j);
                    }

                    cur = data.get(i);
                    countsLeft[cur.cl - 1]++;
                    countsRight[cur.cl - 1]--;
                }
            }

            return new Node(bestRuleInd, bestRuleValue, trainImpl(bestLeft, h + 1), trainImpl(bestRight, h + 1));
        }

        int H(Node node, int n) {
            if (node.isLeaf) {
                return n + 1;
            }
            return Math.max(H(node.left, n + 1), H(node.right, n + 1));
        }

        void train(List<Item> data) throws Exception {
            root = trainImpl(data, 1);
            if (H(root, 0) > 11)
                throw new Exception();
        }

        void print() {
            System.out.println(treeSize(root));
            toPrints = new ArrayList<>();
            collectTree(root, 1);
            for (toPrint i : toPrints) {
                if (i.isLeaf)
                    System.out.println("C " + i.a);
                else
                    System.out.println("Q " + i.a + " " + i.b + " " + i.c + " " + i.d);
            }
        }

        private int treeSize(Node v) {
            if (v.isLeaf)
                return 1;
            return (treeSize(v.left) + treeSize(v.right) + 1);
        }

        class toPrint {
            boolean isLeaf;
            int a, c, d;
            double b;
            toPrint(boolean isLeaf, int a, double b, int c, int d) {
                this.isLeaf = isLeaf;
                this.a = a;
                this.b = b;
                this.c = c;
                this.d = d;
            }
        }

        List<toPrint> toPrints;

        int collectTree(Node v, int n) {
            if (v.isLeaf) {
                toPrints.add(new toPrint(true, (v.ruleInd + 1), 0, 0, 0));
                return n;
            } else {
                int pos = toPrints.size();
                int l = collectTree(v.left, n + 1);
                int r = collectTree(v.right, l + 1);
                toPrints.add(pos, new toPrint(false, (v.ruleInd + 1), v.ruleValue + 0.5, n + 1, l + 1));
                return r;
            }
        }
    }

    private static BufferedReader br;
    private static StringTokenizer in;

    static private String nextToken() {
        while (in == null || !in.hasMoreTokens()) {
            try {
                in = new StringTokenizer(br.readLine());
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return in.nextToken();
    }

    public static void main(String[] args) throws Exception {
        br = new BufferedReader(new InputStreamReader(System.in));
        int m = Integer.parseInt(nextToken());
        int k = Integer.parseInt(nextToken());
        int n = Integer.parseInt(nextToken());

        List<Item> trainItems = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            List<Integer> f = new ArrayList<>();
            for (int j = 0; j < m; j++) {
                f.add(Integer.parseInt(nextToken()));
            }
            int label = Integer.parseInt(nextToken());
            trainItems.add(new Item(f, label));
        }

        DecisionTree tree = new DecisionTree(n, m, k);
        tree.train(trainItems);
        tree.print();
    }

}

