package com.contextmesh;

public class SetupConfig {
    public String inputPath;
    public String outputPath;
    public boolean reloadContext;
    public boolean depthSearch;

    SetupConfig(String inputPath, String outputPath, boolean reloadContext, boolean depthSearch) {
        this.inputPath = inputPath;
        this.outputPath = outputPath;
        this.reloadContext = reloadContext;
        this.depthSearch = depthSearch;
    }
}
