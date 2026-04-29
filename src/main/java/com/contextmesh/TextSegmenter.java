package com.contextmesh;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class TextSegmenter {
    private final String content;

    TextSegmenter(String content) {
        this.content = content;
    }

    public String[] getParagraphs() {
        return content.split("\\n\\s*\\n");
    }

    public String[] getPages() {
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
}
