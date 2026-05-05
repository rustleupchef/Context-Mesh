package com.contextmesh;

public class DocumentPayload {
    public String path;
    public String basePath;
    public float score;

    DocumentPayload(String path, String basePath, float score) {
        this.path = path;
        this.basePath = basePath;
        this.score = score;
    }
}