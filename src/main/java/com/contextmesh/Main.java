package com.contextmesh;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Map;
import java.util.Scanner;
import java.util.UUID;
import java.util.concurrent.Executors;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.tika.Tika;
import org.apache.tika.exception.TikaException;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.sun.net.httpserver.HttpServer;

import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

class Paired {
    public String path;
    public String basePath;
    public String text;

    Paired(String path, String basePath, String text) {
        this.path = path;
        this.basePath = basePath;
        this.text = text;
    }
}

class PromptPayload {
    public String prompt;

    PromptPayload(String prompt) {
        this.prompt = prompt;
    }
}

class MessagePayload {
    public String type;
    public String content;

    MessagePayload(String type, String content) {
        this.type = type;
        this.content = content;
    }
}

class DocumentPayload {
    public String path;
    public String basePath;
    public float score;

    DocumentPayload(String path, String basePath, float score) {
        this.path = path;
        this.basePath = basePath;
        this.score = score;
    }
}

class SetupConfig {
    public String inputPath;
    public String outputPath;
    public boolean reloadContext;

    SetupConfig(String inputPath, String outputPath, boolean reloadContext) {
        this.inputPath = inputPath;
        this.outputPath = outputPath;
        this.reloadContext = reloadContext;
    }
}

public class Main {

    private static Directory getDirectory(String outputPath) throws IOException {
        File directory = Path.of(outputPath).resolve(".index/").toFile();
        directory.mkdirs();

        return FSDirectory.open(Path.of(outputPath).resolve(".index/"));
    }

    private static boolean isTextBased(String mimeType) {
        return mimeType.startsWith("text/")
                || mimeType.equals("application/json")
                || mimeType.equals("application/xml")
                || mimeType.equals("application/xhtml+xml")
                || mimeType.equals("application/javascript")
                || mimeType.equals("application/x-sh")
                || mimeType.contains("wordprocessingml")
                || mimeType.contains("spreadsheetml")
                || mimeType.contains("presentationml")
                || mimeType.equals("application/pdf")
                || mimeType.equals("application/rtf");
    }

    private static String milliSecondsToTime(long ms) {
        long seconds = ms / 1000;
        long minutes = seconds / 60;
        long hours = minutes / 60;

        seconds %= 60;
        minutes %= 60;

        return String.format("%02d:%02d:%02d", hours, minutes, seconds);
    }

    private static void printProgress(int current, int total, long startTime) {
        int percent = (current * 100) / total;
        int barWidth = 20; // total characters in the bar
        int completed = (current * barWidth) / total;

        StringBuilder bar = new StringBuilder("[");
        for (int i = 0; i < barWidth; i++) {
            if (i < completed) bar.append("=");
            else bar.append(" ");
        }
        bar.append("] " + percent + "%");
        System.out.print("\r" + bar.toString() + "\t" + milliSecondsToTime(System.currentTimeMillis() - startTime));
    }

    private static void clearDirectory(File directory) {
        File[] files = directory.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isDirectory()) {
                    clearDirectory(file);
                }
                file.delete();
            }
        }
    }

    private static ArrayList<DocumentPayload> prompt(
        Directory directory, 
        Predictor<String, float[]> embedder, 
        String text, 
        int size
    ) throws IOException {
        DirectoryReader reader = DirectoryReader.open(directory);
        IndexSearcher searcher = new IndexSearcher(reader);

        KnnFloatVectorQuery query = null;
        try {
            query = new KnnFloatVectorQuery("embedding", embedder.predict(text), 10);
        } catch (TranslateException e) {
            e.printStackTrace();
        }
        TopDocs results = searcher.search(query, size);
        StoredFields storedFields = searcher.storedFields();

        ArrayList<DocumentPayload> responsePayload = new ArrayList<>();
        HashSet<String> uniquePaths = new HashSet<>();
        for (ScoreDoc doc : results.scoreDocs) {
            Document document = storedFields.document(doc.doc);
            String path = document.get("basePath");

            if (!uniquePaths.contains(path)) {
                uniquePaths.add(path);
            }

            DocumentPayload docPayload = new DocumentPayload(
                document.get("path"),
                document.get("basePath"),
                doc.score
            );
            responsePayload.add(docPayload);
        }

        reader.close();
        return responsePayload;
    }

    private static void loadContext(
        File[] contextFiles,
        String output_path, 
        Tika tika, 
        ArrayList<Paired> pairs, 
        Predictor<String, float[]> embedder,
        Directory directory
    ) throws IOException {

        clearDirectory(Path.of(output_path).resolve("input").toFile());

        System.out.println("Processing files...");
        int _current = 0, _total = contextFiles.length;
        long _startTime = System.currentTimeMillis();
        for (File file : contextFiles) {
            printProgress(_current, _total, _startTime);

            String mimeType = tika.detect(file);
            if (!isTextBased(mimeType))
                continue;

            String baseName = UUID.randomUUID().toString();

            String text = "";
            try {
                text = tika.parseToString(file).toLowerCase();
            } catch (TikaException e) {
                e.printStackTrace();
            }
            File outputFile = Path.of(output_path).resolve("input").resolve(baseName + ".txt").toFile();
            FileWriter _writer = new FileWriter(outputFile);
            _writer.write(text);
            _writer.close();

            pairs.add(new Paired(outputFile.getAbsolutePath(), file.getAbsolutePath(), text));

            _current++;
        }
        printProgress(_current, _total, _startTime);
        System.out.println("\nFinished processing files.\n");


        _startTime = System.currentTimeMillis();
        StandardAnalyzer _analyzer = new StandardAnalyzer();
        IndexWriterConfig _config = new IndexWriterConfig(_analyzer);
        IndexWriter _writer = new IndexWriter(directory, _config);

        System.out.println("\n\nIndexing documents...");
        _current = 0;
        _total = pairs.size();

        for (Paired pair : pairs) {
            printProgress(_current, _total, _startTime);
            float[] vector = new float[0];
            try {
                vector = embedder.predict(pair.text);
            } catch (TranslateException e) {
                e.printStackTrace();
            }

            Document doc = new Document();
            doc.add(new KnnFloatVectorField("embedding", vector, VectorSimilarityFunction.COSINE));
            doc.add(new TextField("content", pair.text, Field.Store.YES));
            doc.add(new TextField("path", pair.path, Field.Store.YES));
            doc.add(new TextField("basePath", pair.basePath, Field.Store.YES));
            _writer.addDocument(doc);

            _current++;
        }
        _writer.commit();
        _writer.close();

        printProgress(_current, _total, _startTime);
        System.out.println("\nFinished indexing documents.\n");
    }


    public static void main(String[] args) throws Exception {
        Gson gson = new Gson();
        SetupConfig config = null;

        try (FileReader reader = new FileReader("config.json")) {
            config = gson.fromJson(reader, SetupConfig.class);
        } catch (Exception e) {
            System.out.println("Error reading config.json: " + e.getMessage());
            System.exit(1);
        }

        final String input_path = config.inputPath;
        final String output_path = config.outputPath;
        final File inputDir = new File(input_path), outputDir = new File(output_path);

        if (!inputDir.isDirectory() || !outputDir.isDirectory()) {
            System.out.println("Please enter only directories");
            System.exit(1);
        }

        File[] contextFiles = inputDir.listFiles();
        Tika tika = new Tika();

        ArrayList<Paired> pairs = new ArrayList<>();

        File dir = Path.of(output_path).resolve("input/").toFile();
        dir.mkdirs();

        Criteria<String, float[]> criteria = Criteria.builder()
            .setTypes(String.class, float[].class)
            .optModelUrls("djl://ai.djl.huggingface.pytorch/sentence-transformers/all-mpnet-base-v2")
            .optEngine("PyTorch")
            .build();
        
        ZooModel<String, float[]> model = ModelZoo.loadModel(criteria);
        Predictor<String, float[]> embedder = model.newPredictor();

        Directory directory = getDirectory(output_path);


        if (config.reloadContext)
            loadContext(contextFiles, output_path, tika, pairs, embedder, directory);

        HttpServer server = HttpServer.create(new InetSocketAddress(8080), 0);

        server.createContext("/api/prompt", exchange -> {
            if ("POST".equals(exchange.getRequestMethod())) {
                String requestBody = new String(
                    exchange.getRequestBody().readAllBytes(),
                    StandardCharsets.UTF_8
                );

                String text = new Gson().fromJson(requestBody, PromptPayload.class).prompt;
                ArrayList<DocumentPayload> responsePayload = prompt(directory, embedder, text, 10);
                
                exchange.sendResponseHeaders(200, 0);
                exchange.getResponseBody().write(
                    new GsonBuilder()
                        .setPrettyPrinting()
                        .create()
                        .toJson(responsePayload)
                        .getBytes(StandardCharsets.UTF_8)
                );

            } else {
                exchange.sendResponseHeaders(405, -1);
            }
            exchange.close();
        });

        server.createContext("/api/reload_context", exchange -> {
            if ("POST".equals(exchange.getRequestMethod())) {
                MessagePayload responsePayload = new MessagePayload("success", "Context reloaded successfully");
                pairs.clear();
                loadContext(inputDir.listFiles(), output_path, tika, pairs, embedder, directory);

                exchange.sendResponseHeaders(200, 0);
                exchange.getResponseBody().write(
                    new GsonBuilder()
                        .setPrettyPrinting()
                        .create()
                        .toJson(responsePayload)
                        .getBytes(StandardCharsets.UTF_8)
                );
            } else {
                exchange.sendResponseHeaders(405, -1);
            }
            exchange.close();
        });

        server.setExecutor(Executors.newVirtualThreadPerTaskExecutor());
        server.start();

        System.out.println("Server started on port 8080");
    }

}
