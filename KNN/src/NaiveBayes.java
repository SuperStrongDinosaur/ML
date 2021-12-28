import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.StringTokenizer;

public class NaiveBayes {
    public static class Bayes {

        private int numOfAllMessages = 0;
        private int[][] featureMatrix;
        private int[] LabelVector;

        private static BufferedReader br;
        private static StringTokenizer in;

        Bayes() {
            br = new BufferedReader(new InputStreamReader(System.in));
        }

        static HashMap<Integer, Integer> hashSpamWords;
        static HashMap<Integer, Integer> hashLegalWords;

        static int numOfSpamMessages = 0;
        static int numOfSpamWords = 0;
        static int numOfLegalMessages = 0;
        static int numOfLegalWords = 0;
        static int numOfUniqWords = 0;

        private void addWord(HashMap<Integer, Integer> hashWords, int[] words, boolean isSpam) {
            for (int sw : words) {
                if (!hashSpamWords.containsKey(sw) && !hashLegalWords.containsKey(sw)) {
                    numOfUniqWords++;
                }
                if (hashWords.containsKey(sw)) {
                    hashWords.put(sw, hashWords.get(sw) + 1);
                } else {
                    hashWords.put(sw, 1);
                }
                if (isSpam)
                    numOfSpamWords++;
                else
                    numOfLegalWords++;

            }
            if (isSpam)
                numOfSpamMessages++;
            else
                numOfLegalMessages++;
        }

        void train() {
            hashSpamWords = new HashMap<>();
            hashLegalWords = new HashMap<>();

            for (int i = 0; i < numOfAllMessages; i++) {
                if (LabelVector[i] == 0) {
                    addWord(hashSpamWords, featureMatrix[i], true);
                }

                else if (LabelVector[i] == 1) {
                    addWord(hashLegalWords, featureMatrix[i], false);
                }
            }
        }

        void test(int[][] input) {
            for (int[] l : input) {

                double SpamProb = Math.log((double)numOfSpamMessages / numOfAllMessages * 0);
                for (int aL : l) {
                    double pi = 0;
                    if (hashSpamWords.get(aL) != null) {
                        pi = (double) (hashSpamWords.get(aL) + 1) / (numOfSpamWords + numOfUniqWords);
                    } else {
                        pi = (double) (1) / (numOfSpamWords + numOfUniqWords);
                    }
                    SpamProb = SpamProb + Math.log(pi);
                }

                double LegalProb = Math.log((double)numOfLegalMessages / numOfAllMessages   );
                for (int aL : l) {
                    double pi;
                    if (hashLegalWords.get(aL) != null) {
                        pi = (double) (hashLegalWords.get(aL) + 1) / (numOfLegalWords + numOfUniqWords);
                    } else {
                        pi = (double) (1) / (numOfLegalWords + numOfUniqWords);
                    }
                    LegalProb = LegalProb + Math.log(pi);
                }

                //double prob = (double) 1 / ((double)1 + Math.pow(Math.E, (LegalProb - SpamProb)));

                //SpamProb = SpamProb * 0.95;

                //System.out.println(SpamProb);
                //System.out.println(LegalProb);
                //System.out.println(prob);
                if (LegalProb < SpamProb) {
                    System.out.println("S");
                } else {
                    System.out.println("L");
                }

            }
        }

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

        void readData() {
            int n = Integer.parseInt(nextToken());

            featureMatrix = new int[n][];
            LabelVector = new int[n];
            numOfAllMessages = n;

            for (int i = 0; i < n; i++) {
                int m = Integer.parseInt(nextToken());
                featureMatrix[i] = new int[m];

                String labele = nextToken();
                if (labele.contains("S"))
                    LabelVector[i] = 0;
                else
                    LabelVector[i] = 1;

                for (int j = 0; j < m; j++) {
                    featureMatrix[i][j] = Integer.parseInt(nextToken());
                }
            }
        }
    }

    public static void main(String[] args) {
        Bayes nb = new Bayes();
        nb.readData();
        nb.train();

        int n = Integer.parseInt(Bayes.nextToken());
        int[][] featureMatrix = new int[n][];
        for (int i = 0; i < n; i++) {
            int m = Integer.parseInt(Bayes.nextToken());
            featureMatrix[i] = new int[m];

            for (int j = 0; j < m; j++) {
                featureMatrix[i][j] = Integer.parseInt(Bayes.nextToken());
            }
        }
        nb.test(featureMatrix);
    }

}