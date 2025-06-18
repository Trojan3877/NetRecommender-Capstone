package recommender.java;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Collaborative Filtering Recommender
 * Uses Pearson correlation between users to generate recommendations.
 */
public class CollaborativeFiltering {

    private Map<Integer, Map<Integer, Double>> userRatings;

    public CollaborativeFiltering(Map<Integer, Map<Integer, Double>> userRatings) {
        this.userRatings = userRatings;
    }

    /**
     * Computes Pearson correlation between two users.
     */
    private double pearsonCorrelation(Map<Integer, Double> ratings1, Map<Integer, Double> ratings2) {
        Set<Integer> common = new HashSet<>(ratings1.keySet());
        common.retainAll(ratings2.keySet());

        if (common.isEmpty()) return 0.0;

        double sum1 = 0, sum2 = 0, sum1Sq = 0, sum2Sq = 0, pSum = 0;
        for (Integer movieId : common) {
            double r1 = ratings1.get(movieId);
            double r2 = ratings2.get(movieId);

            sum1 += r1;
            sum2 += r2;
            sum1Sq += r1 * r1;
            sum2Sq += r2 * r2;
            pSum += r1 * r2;
        }

        int n = common.size();
        double numerator = pSum - (sum1 * sum2 / n);
        double denominator = Math.sqrt((sum1Sq - (sum1 * sum1 / n)) * (sum2Sq - (sum2 * sum2 / n)));

        return denominator == 0 ? 0 : numerator / denominator;
    }

    /**
     * Recommend movies for a given user based on similar users' ratings.
     */
    public List<Integer> recommend(int userId, int topN) {
        Map<Integer, Double> targetRatings = userRatings.get(userId);
        Map<Integer, Double> scores = new HashMap<>();

        for (Map.Entry<Integer, Map<Integer, Double>> entry : userRatings.entrySet()) {
            int otherUser = entry.getKey();
            if (otherUser == userId) continue;

            double similarity = pearsonCorrelation(targetRatings, entry.getValue());
            for (Map.Entry<Integer, Double> ratingEntry : entry.getValue().entrySet()) {
                int movieId = ratingEntry.getKey();
                if (!targetRatings.containsKey(movieId)) {
                    scores.put(movieId, scores.getOrDefault(movieId, 0.0) + similarity * ratingEntry.getValue());
                }
            }
        }

        return scores.entrySet().stream()
                .sorted((a, b) -> Double.compare(b.getValue(), a.getValue()))
                .limit(topN)
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
    }
}
