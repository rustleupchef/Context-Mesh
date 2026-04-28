package com.contextmesh;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class TextSegmenter {
    private final String content;
    private final String[] lines;

    TextSegmenter(String content) {
        this.content = content;
        this.lines = content.split("\\r?\\n");
    }

    public String[] getParagraphs() {
        return content.split("\\n\\s*\\n");
    }

    public String[] getPages() {
        return content.split("(?<=[.!?])\\s+");
    }

    public String[] getSentences() {
        return content.split("(?<=[.!?])\\s+");
    }

    public Map<String, String> getSections() {
        Pattern header = Pattern.compile(
            "^([A-Z][A-Z\\s]{2,}|#{1,6}\\s.+|\\d+\\.\\s+[A-Z].+)$",
            Pattern.MULTILINE
        );

        Map<String, String> sections = new LinkedHashMap<>();
        String[] parts = header.split(content);
        Matcher hm = header.matcher(content);

        List<String> headers = new ArrayList<>();
        while (hm.find()) headers.add(hm.group().strip());

        for (int i = 0; i < headers.size(); i++) {
            String body = (i < parts.length - 1) ? parts[i + 1].strip() : "";
            sections.put(headers.get(i), body);
        }
        return sections;
    }

    public List<String> extractKeyInfo(int topN) {
        String[] sentences = this.getSentences();
        
        Map<String, Long> docFreq = new HashMap<>();
        List<Map<String, Long>> sentenceTerms = new ArrayList<>();

        for (String sent : sentences) {
            Map<String, Long> tf = Arrays.stream(sent.toLowerCase().split("\\W+"))
                .filter(w -> w.length() > 3) // skip short words
                .collect(Collectors.groupingBy(w -> w, Collectors.counting()));
            sentenceTerms.add(tf);
            tf.keySet().forEach(w -> docFreq.merge(w, 1L, Long::sum));
        }

        int N = sentences.length;
        double[] scores = new double[N];
        for (int i = 0; i < N; i++) {
            for (Map.Entry<String, Long> e : sentenceTerms.get(i).entrySet()) {
                double tf = e.getValue();
                double idf = Math.log((double) N / (docFreq.get(e.getKey()) + 1));
                scores[i] += tf * idf;
            }
        }

        Integer[] idx = IntStream.range(0, N).boxed().toArray(Integer[]::new);
        Arrays.sort(idx, (a, b) -> Double.compare(scores[b], scores[a]));
        
        return Arrays.stream(Arrays.copyOf(idx, topN))
            .sorted()
            .map(i -> sentences[i])
            .collect(Collectors.toList());
    }
}
