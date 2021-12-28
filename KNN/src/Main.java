import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.*;
import java.util.function.BiConsumer;
import java.util.function.DoubleUnaryOperator;

public class Main {
    static class Obj {
        double[] attributes;
        int classNum;
        int i;

        Obj(double[] attributes, int classNum, int i) {
            this.classNum = classNum;
            this.attributes = attributes;
            this.i = i;
        }

        Obj(Obj element) {
            this.i = element.i;
            this.classNum = element.classNum;
            this.attributes = new double[element.attributes.length];
            System.arraycopy(element.attributes, 0, this.attributes, 0, element.attributes.length);
        }
    }

    private interface DistanceMetric {
        double calc(Obj a, Obj b);
    }

    private static final class EuclideanMetric implements DistanceMetric {
        @Override
        public double calc(Obj a, Obj b) {
            double sum = 0;
            for (int i = 0; i < a.attributes.length; i++) {
                sum += Math.pow(a.attributes[i] - b.attributes[i], 2);
            }
            return Math.sqrt(sum);
        }
    }

    private static final class ManhattanDistance implements DistanceMetric {
        @Override
        public double calc(Obj a, Obj b) {
            double sum = 0;
            for (int i = 0; i < a.attributes.length; i++) {
                sum += Math.abs(a.attributes[i] - b.attributes[i]);
            }
            return sum;
        }
    }

    private static final class ChebyshevDistance implements DistanceMetric {
        @Override
        public double calc(Obj a, Obj b) {
            double max = -1.;
            for (int i = 0; i < a.attributes.length; i++) {
                max = Math.max(Math.abs(a.attributes[i] - b.attributes[i]), max);
            }
            return max;
        }
    }

    private static double uniformKernel(double u) {
        if (Math.abs(u) > 1) return 0;
        return 0.5;
    }

    private static double triangularKernel(double u) {
        if (Math.abs(u) > 1) return 0;
        return 1. -  Math.abs(u);
    }

    private static double epanechnikovKernel(double u) {
        if (Math.abs(u) > 1) return 0;
        return 3. / 4. * (1. - u * u);
    }

    protected static final class KNNClassifier {
        private final DistanceMetric metricFunction;
        private final DoubleUnaryOperator kernelFunction;
        private final int k;
        private final Obj X[];
        private final int y[];
        private final int cnt;

        int getAmountOfNeighbours() {
            return k;
        }

        void reportAllNeighbours(Obj u, int cnt, BiConsumer<Obj, Double> callback) {
            Obj[] sortedUNeighbours = sort(u);
            for (int i = 0; i < cnt; i++) {
                Obj x = sortedUNeighbours[i];
                double w = calcWeight(u, x, sortedUNeighbours);
                callback.accept(x, w);
            }
            if (cnt == 0) {
                callback.accept(sortedUNeighbours[0], 0.);
            }
        }

        double calcWeight(Obj u, Obj x, Obj[] obj) {
            if ((k + 1) >= obj.length) {
                return 0;
            }
            return kernelFunction.applyAsDouble(metricFunction.calc(u, x) / metricFunction.calc(u, obj[k + 1]));
        }

        private Obj[] sort(final Obj u) {
            Arrays.sort(X, Comparator.comparingDouble(o -> metricFunction.calc(o, u)));
            return X;
        }

        int predict(Obj u) {
            int begI;
            begI = 1;
            double bestAns = -1.;
            int bestClassId = -1;
            Obj[] sorted = sort(u);
            for (int i = 1; i <= cnt; i++) {
                double ans = 0;
                for (int j = begI; j < X.length; j++) {
                    Obj x = sorted[j];
                    if (y[x.classNum] == i) {
                        ans += calcWeight(u, x, sorted);
                    }
                }
                if (ans > bestAns) {
                    bestAns = ans;
                    bestClassId = i;
                }
            }
            return bestClassId;
        }

        KNNClassifier(DistanceMetric metricFunction, DoubleUnaryOperator kernelFunction, int k, Obj[] x, int[] y, int cnt) {
            this.metricFunction = metricFunction;
            this.kernelFunction = kernelFunction;
            this.k = k;
            this.X = x;
            this.y = y;
            this.cnt = cnt;
        }
    }

    private static final class F1Evaluator {
        private double safeDivision(double a, double b) {
            if (b == 0) return 0;
            return a / b;
        }

        private class ClassSummary {
            int cnt = 0;
            int rightCnt = 0;
            private Double precision = null;
            private Double recall = null;

            Double getPrecision() {
                if (precision == null) {
                    precision = safeDivision(rightCnt, cnt);
                }
                return precision;
            }

            Double getRecall() {
                if (recall == null) {
                    recall = safeDivision(rightCnt, cnt);
                }
                return recall;
            }
        }

        private int allElementsAmount = 0;
        private ClassSummary[] classSummaries;
        private Double f1Micro = null;

        F1Evaluator(int amountOfClasses) {
            classSummaries = new ClassSummary[amountOfClasses + 1];
            for (int i = 0; i < classSummaries.length; i++) {
                classSummaries[i] = new ClassSummary();
            }
        }

        void addResult(int expected, int predicted) {
            allElementsAmount++;
            if (expected == predicted) {
                ClassSummary classSummary = classSummaries[expected];
                classSummary.cnt++;
                classSummary.rightCnt++;
            } else {
                classSummaries[predicted].cnt++;
            }
        }

        Double reportF1() {
            if (f1Micro != null) return f1Micro;
            double precision = 0.;
            double recall = 0.;
            for (ClassSummary classSummary : classSummaries) {
                precision += classSummary.getPrecision() * (double)classSummary.cnt;
                recall += classSummary.getRecall() * (double)classSummary.cnt;
            }
            precision = safeDivision(precision, (double)allElementsAmount);
            recall = safeDivision(recall, (double)allElementsAmount);
            return safeDivision(2 * (precision * recall), (precision + recall));
        }
    }

    private static Obj[] normalizeAttributes(Obj[] X) {
        int m = X[0].attributes.length;
        int n = X.length;
        Obj[] elements = Arrays.stream(X).map(Obj::new).toArray(Obj[]::new);
        for (int i = 0; i < m; i++) {
            double max = Double.MIN_VALUE;
            double min = Double.MAX_VALUE;

            for (int j = 0; j != n; j++) {
                max = Math.max(elements[j].attributes[i], max);
                min = Math.min(elements[j].attributes[i], min);
            }
            double d = max - min;
            if (d == 0) {
                for (Obj element : elements) {
                    element.attributes[i] = 0;
                }
            } else {
                for (Obj element : elements) {
                    element.attributes[i] = (element.attributes[i] - min) / d;
                }
            }
        }
        return elements;
    }

    private static BufferedReader br;
    private static StringTokenizer in;
    private static PrintWriter out;

    static private String nextToken()  {
        while (in == null || !in.hasMoreTokens()) {
            try {
                in = new StringTokenizer(br.readLine());
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return in.nextToken();
    }

    public static void main(String args[]) {
        br = new BufferedReader(new InputStreamReader(System.in));
        out = new PrintWriter(System.out);

        int M = Integer.parseInt(nextToken());
        int K = Integer.parseInt(nextToken());
        int N = Integer.parseInt(nextToken());

        Obj[] attributesTrain = new Obj[N];
        int[] labelsTrain = new int[N];
        for (int i = 0; i < N; i++) {
            double[] attributes = new double[M];
            for (int j = 0; j < M; j++) {
                attributes[j] = Integer.parseInt(nextToken());
            }
            int c = Integer.parseInt(nextToken());
            attributesTrain[i] = new Obj(attributes, c, i);
            labelsTrain[i] = c;
        }

        int Q = Integer.parseInt(nextToken());
        Obj[] attributesTest = new Obj[Q];
        for (int i = 0; i < Q; i++) {
            double[] attributes = new double[M];
            for (int j = 0; j < M; j++) {
                attributes[j] = Integer.parseInt(nextToken());
            }
            attributesTest[i] = new Obj(attributes, -1, i);
        }

        ArrayList<DistanceMetric> metrics = new ArrayList<>();
        metrics.add(new EuclideanMetric());
        metrics.add(new ManhattanDistance());
        metrics.add(new ChebyshevDistance());

        ArrayList<DoubleUnaryOperator> kernels = new ArrayList<>();
        kernels.add(Main::uniformKernel);
        kernels.add(Main::triangularKernel);
        kernels.add(Main::epanechnikovKernel);

        KNNClassifier bestParams = null;
        Obj[] attributesTrainNorm = normalizeAttributes(attributesTrain);
        double bestReport = Integer.MAX_VALUE;
        for (DistanceMetric metric : metrics) {
            for (DoubleUnaryOperator kernel : kernels) {
                for (int k = 1; k <= Math.min(attributesTrainNorm.length, 20); k++) {
                    KNNClassifier estimator = new KNNClassifier(metric, kernel, k, attributesTrainNorm, labelsTrain, K);

                    F1Evaluator evaluator = new F1Evaluator(K);
                    for (int i = 0; i < attributesTrainNorm.length; i++) {
                        int predicted = estimator.predict(attributesTrainNorm[i]);
                        evaluator.addResult(labelsTrain[i], predicted);
                    }
                    double curReport = evaluator.reportF1();
                    if (Math.abs(1. - curReport) < Math.abs(1. - bestReport)) {
                        bestReport = curReport;
                        bestParams = estimator;
                    }
                }
            }
        }

        Obj[] attributesTestNorm = normalizeAttributes(attributesTest);
        for (Obj anAttributesTest : attributesTestNorm) {
            int neighbours = bestParams.getAmountOfNeighbours();
            out.print(neighbours);
            out.print(" ");
            bestParams.reportAllNeighbours(anAttributesTest, neighbours, (element, weight) -> {
                out.print(element.i + 1);
                out.print(" ");
                out.print(String.format("%.20f", weight));
                out.print(" ");
            });
            out.println();
        }
        out.close();
    }
}
