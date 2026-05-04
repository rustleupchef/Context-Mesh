package com.contextmesh;

import java.io.File;
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


    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            args = new String[2];
            System.out.println("Schema; java -jar [jar_name] [input_path] [output_path] [query]");

            Scanner scanner = new Scanner(System.in);

            System.out.print("Input Path: ");
            args[0] = scanner.nextLine();

            System.out.print("Output Path: ");
            args[1] = scanner.nextLine();

            scanner.close();
        }

        final String input_path = args[0], output_path = args[1];
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

        System.out.println("Processing files...");
        int current = 0, total = contextFiles.length;
        long startTime = System.currentTimeMillis();
        for (File file : contextFiles) {
            printProgress(current, total, startTime);
            
            String mimeType = tika.detect(file);
            if (!isTextBased(mimeType))
                continue;

            String baseName = UUID.randomUUID().toString();

            String text = tika.parseToString(file).toLowerCase();
            File outputFile = Path.of(output_path).resolve("input").resolve(baseName + ".txt").toFile();
            FileWriter writer = new FileWriter(outputFile);
            writer.write(text);
            writer.close();

            pairs.add(new Paired(outputFile.getAbsolutePath(), file.getAbsolutePath(), text));

            TextSegmenter segmenter = new TextSegmenter(text);
            final int minLength = 200;

            // Paragraphs Segmentation
            String[] paragraphs = segmenter.getParagraphs();
            if (paragraphs.length > 0) {
                for (String para : paragraphs) {

                    if (para.strip().isEmpty()) continue;
                    if (para.length() < minLength) continue;

                    baseName = UUID.randomUUID().toString();
                    File paraFile = Path.of(output_path).resolve("input").resolve(baseName + ".txt").toFile();
                    FileWriter paraWriter = new FileWriter(paraFile);
                    paraWriter.write(para);
                    paraWriter.close();
                    pairs.add(new Paired(paraFile.getAbsolutePath(), file.getAbsolutePath(), para));
                }
            }

            // Pages Segmentation
            String[] pages = segmenter.getPages();
            if (pages.length > 0) {
                for (String page : pages) {

                    if (page.strip().isEmpty()) continue;
                    if (page.length() < minLength) continue;

                    baseName = UUID.randomUUID().toString();
                    File pageFile = Path.of(output_path).resolve("input").resolve(baseName + ".txt").toFile();
                    FileWriter pageWriter = new FileWriter(pageFile);
                    pageWriter.write(page);
                    pageWriter.close();
                    pairs.add(new Paired(pageFile.getAbsolutePath(), file.getAbsolutePath(), page));
                }
            }

            // Sections Segmentation
            Map<String, String> sections = segmenter.getSections();
            if (sections.size() > 0) {
                for (Map.Entry<String, String> entry : sections.entrySet()) {
                    String header = entry.getKey();
                    String body = entry.getValue();
                    String hbText = header + "\n" + body;
                    
                    if (hbText.strip().isEmpty()) continue;
                    if (hbText.length() < minLength) continue;

                    baseName = UUID.randomUUID().toString();
                    File sectionFile = Path.of(output_path).resolve("input").resolve(baseName + ".txt").toFile();
                    FileWriter sectionWriter = new FileWriter(sectionFile);
                    sectionWriter.write(header + "\n" + body);
                    sectionWriter.close();
                    pairs.add(new Paired(sectionFile.getAbsolutePath(), file.getAbsolutePath(), hbText));
                }
            }
            
            current++;
        }
        printProgress(current, total, startTime);
        System.out.println("\nFinished processing files.\n");

        Criteria<String, float[]> criteria = Criteria.builder()
            .setTypes(String.class, float[].class)
            .optModelUrls("djl://ai.djl.huggingface.pytorch/sentence-transformers/all-mpnet-base-v2")
            .optEngine("PyTorch")
            .build();
        
        ZooModel<String, float[]> model = ModelZoo.loadModel(criteria);
        Predictor<String, float[]> embedder = model.newPredictor();

        Directory directory = getDirectory(output_path);

        startTime = System.currentTimeMillis();
        StandardAnalyzer analyzer = new StandardAnalyzer();
        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        IndexWriter writer = new IndexWriter(directory, config);


        System.out.println("\n\nIndexing documents...");
        current = 0;
        total = pairs.size();

        for (Paired pair : pairs) {
            printProgress(current, total, startTime);
            float[] vector = embedder.predict(pair.text);

            Document doc = new Document();
            doc.add(new KnnFloatVectorField("embedding", vector, VectorSimilarityFunction.COSINE));
            doc.add(new TextField("content", pair.text, Field.Store.YES));
            doc.add(new TextField("path", pair.path, Field.Store.YES));
            doc.add(new TextField("basePath", pair.basePath, Field.Store.YES));
            writer.addDocument(doc);

            current++;
        }
        writer.commit();
        writer.close();

        printProgress(current, total, startTime);
        System.out.println("\nFinished indexing documents.\n");

        HttpServer server = HttpServer.create(new InetSocketAddress(8080), 0);

        server.createContext("/api/prompt", exchange -> {
            if ("POST".equals(exchange.getRequestMethod())) {
                String requestBody = new String(
                    exchange.getRequestBody().readAllBytes(),
                    StandardCharsets.UTF_8
                );

                DirectoryReader reader = DirectoryReader.open(directory);
                IndexSearcher searcher = new IndexSearcher(reader);

                PromptPayload payload = new Gson().fromJson(requestBody, PromptPayload.class);
                KnnFloatVectorQuery query = null;
                try {
                    query = new KnnFloatVectorQuery("embedding", embedder.predict(payload.prompt), 10);
                } catch (TranslateException e) {
                    e.printStackTrace();
                }
                TopDocs results = searcher.search(query, pairs.size());
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

                File[] _contextFiles = inputDir.listFiles();

                System.out.println("Processing files...");
                int _current = 0, _total = _contextFiles.length;
                long _startTime = System.currentTimeMillis();
                for (File file : _contextFiles) {
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

                    TextSegmenter segmenter = new TextSegmenter(text);
                    final int minLength = 200;

                    // Paragraphs Segmentation
                    String[] paragraphs = segmenter.getParagraphs();
                    if (paragraphs.length > 0) {
                        for (String para : paragraphs) {

                            if (para.strip().isEmpty()) continue;
                            if (para.length() < minLength) continue;

                            baseName = UUID.randomUUID().toString();
                            File paraFile = Path.of(output_path).resolve("input").resolve(baseName + ".txt").toFile();
                            FileWriter paraWriter = new FileWriter(paraFile);
                            paraWriter.write(para);
                            paraWriter.close();
                            pairs.add(new Paired(paraFile.getAbsolutePath(), file.getAbsolutePath(), para));
                        }
                    }

                    // Pages Segmentation
                    String[] pages = segmenter.getPages();
                    if (pages.length > 0) {
                        for (String page : pages) {

                            if (page.strip().isEmpty()) continue;
                            if (page.length() < minLength) continue;

                            baseName = UUID.randomUUID().toString();
                            File pageFile = Path.of(output_path).resolve("input").resolve(baseName + ".txt").toFile();
                            FileWriter pageWriter = new FileWriter(pageFile);
                            pageWriter.write(page);
                            pageWriter.close();
                            pairs.add(new Paired(pageFile.getAbsolutePath(), file.getAbsolutePath(), page));
                        }
                    }

                    // Sections Segmentation
                    Map<String, String> sections = segmenter.getSections();
                    if (sections.size() > 0) {
                        for (Map.Entry<String, String> entry : sections.entrySet()) {
                            String header = entry.getKey();
                            String body = entry.getValue();
                            String hbText = header + "\n" + body;
                            
                            if (hbText.strip().isEmpty()) continue;
                            if (hbText.length() < minLength) continue;

                            baseName = UUID.randomUUID().toString();
                            File sectionFile = Path.of(output_path).resolve("input").resolve(baseName + ".txt").toFile();
                            FileWriter sectionWriter = new FileWriter(sectionFile);
                            sectionWriter.write(header + "\n" + body);
                            sectionWriter.close();
                            pairs.add(new Paired(sectionFile.getAbsolutePath(), file.getAbsolutePath(), hbText));
                        }
                    }
                    
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
