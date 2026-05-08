package com.contextmesh;

public class SetupConfig {

    public String inputPath;
    public String outputPath;
    public boolean reloadContext;
    public boolean depthSearch;
    public int minChunkSize;
    public int port;
    public boolean append;

    SetupConfig(
            String inputPath,
            String outputPath,
            boolean reloadContext,
            boolean depthSearch,
            int minChunkSize,
            int port,
            boolean append) {
        this.inputPath = inputPath;
        this.outputPath = outputPath;
        this.reloadContext = reloadContext;
        this.depthSearch = depthSearch;
        this.minChunkSize = minChunkSize;
        this.port = port;
        this.append = append;
    }
}
