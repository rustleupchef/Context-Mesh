package com.contextmesh;

public class SetupConfig {
    public String inputPath;
    public String outputPath;
    public boolean reloadContext;

    SetupConfig(String inputPath, String outputPath, boolean reloadContext) {
        this.inputPath = inputPath;
        this.outputPath = outputPath;
        this.reloadContext = reloadContext;
    }
}
