package recommender.java;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Content-Based Filtering Recommender
 * Recommends movies similar to user's preferred genres using TF-IDF-style matching.
 */
public class ContentBasedFiltering {

    private Map<Integer, String[]> movieGenres; // movieId -> array of genres
    private Map<Integer, Set<Integer>> userWatched;

    public ContentBasedFiltering(Map<Integer, String[]> movieGenres, Map<Integer, Set<Integer>> userWatched) {
        this.movieGenres = movieGenres;
        this.userWatched = userWatched;
    }

    /**
     * Converts genres to a frequency map.
     */
    private Map<String, Integer> buildGenreVector(String[] genres) {
        Map<String, Integer> vector = new HashMap<>();
        for (String genre : genres) {
            vector.put(genre, vector.getOrDefault(genre, 0) + 1);
        }
        return vector;
    }

    /**
     * Computes cosine similarity between two genre vectors.
     */
    private double cosineSimilarity(Map<String, Integer> v1, Map<String, Integer> v2) {
        Set<String> allGenres = new HashSet<>(v1.keySet());
        allGenres.addAll(v2.keySet());

        double dot = 0, mag1 = 0, mag2 = 0;
        for (String genre : allGenres) {
            int g1 = v1.getOrDefault(genre, 0);
            int g2 = v2.getOrDefault(genre, 0);
            dot += g1 * g2;
            mag1 += g1 * g1;
            mag2 += g2 * g2;
        }

        return mag1 == 0 || mag2 == 0 ? 0 : dot / (Math.sqrt(mag1) * Math.sqrt(mag2));
    }

    /**
     * Recommends top N movies similar to user's watched genres.
     */
    public List<Integer> recommend(int userId, int topN) {
        Set<Integer> watched = userWatched.getOrDefault(userId, new HashSet<>());
        Map<String, Integer> userGenreVector = new HashMap<>();

        // Aggregate genres from watched movies
        for (int movieId : watched) {
            String[] genres = movieGenres.get(movieId);
            if (genres != null) {
                for (String genre : genres) {
                    userGenreVector.put(genre, userGenreVector.getOrDefault(genre, 0) + 1);
                }
            }
        }

        Map<Integer, Double> similarityScores = new HashMap<>();
        for (Map.Entry<Integer, String[]> entry : movieGenres.entrySet()) {
            int movieId = entry.getKey();
            if (watched.contains(movieId)) continue;

            Map<String, Integer> genreVector = buildGenreVector(entry.getValue());
            double similarity = cosineSimilarity(userGenreVector, genreVector);
            similarityScores.put(movieId, similarity);
        }

        return similarityScores.entrySet().stream()
                .sorted((a, b) -> Double.compare(b.getValue(), a.getValue()))
                .limit(topN)
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
    }
}
