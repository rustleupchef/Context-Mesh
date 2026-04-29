package com.contextmesh;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Map;
import java.util.Scanner;
import java.util.UUID;

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

import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;

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
        if (args.length < 3) {
            args = new String[3];
            System.out.println("Schema; java -jar [jar_name] [input_path] [output_path] [query]");

            Scanner scanner = new Scanner(System.in);

            System.out.print("Input Path: ");
            args[0] = scanner.nextLine();

            System.out.print("Output Path: ");
            args[1] = scanner.nextLine();

            System.out.print("Query: ");
            args[2] = scanner.nextLine().toLowerCase();

            scanner.close();
        }

        final String input_path = args[0], output_path = args[1];
        final File inputDir = new File(input_path), outputDir = new File(output_path);
        final String prompt = args[2];

        if (!inputDir.isDirectory() || !outputDir.isDirectory()) {
            System.out.println("Please enter only directories");
            System.exit(1);
        }

        File[] contextFiles = inputDir.listFiles();
        Tika tika = new Tika();

        ArrayList<Paired> pairs = new ArrayList<>();

        File dir = Path.of(output_path).resolve("input/").toFile();
        ProcessBuilder builder = new ProcessBuilder();
        Process process;
        builder.directory(dir);

        if (dir.mkdirs()) {
            builder.command("git", "init");
            process = builder.start();
            process.waitFor();
        }

        builder.command("bash", "-c", "ls -a | grep .git");
        process = builder.start();
        process.waitFor();

        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        if (bufferedReader.readLine() == null) {
            builder.command("git", "init");
            process = builder.start();
            process.waitFor();
        }

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

        builder.command("git", "diff");
        process = builder.start();
        process.waitFor();
        bufferedReader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        String output1 = bufferedReader.readLine();

        builder.command("git", "log");
        process = builder.start();
        process.waitFor();
        bufferedReader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        String output2 = bufferedReader.readLine();

        startTime = System.currentTimeMillis();
        if (output1 != null || output2 == null) {
            builder.command("git", "add", ".");
            process = builder.start();
            process.waitFor();

            builder.command("git", "commit", "-m", "\"change\"");
            process = builder.start();
            process.waitFor();

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
        } else {
            System.out.println("No changes detected. Skipping indexing.");
            System.out.println(milliSecondsToTime(System.currentTimeMillis() - startTime) + "\n\n");
        }

        DirectoryReader reader = DirectoryReader.open(directory);
        IndexSearcher searcher = new IndexSearcher(reader);

        KnnFloatVectorQuery query = new KnnFloatVectorQuery("embedding", embedder.predict(prompt), 10);
        TopDocs results = searcher.search(query, pairs.size());
        StoredFields storedFields = searcher.storedFields();

        HashSet<String> uniquePaths = new HashSet<>();
        for (ScoreDoc doc : results.scoreDocs) {
            Document document = storedFields.document(doc.doc);
            String path = document.get("basePath");

            if (!uniquePaths.contains(path)) {
                uniquePaths.add(path);
            }

            for (int i = 0; i < 30; i++) {
                System.out.print("=");
            }
            System.out.println();

            System.out.println("Base Path: " + path);
            System.out.println("Path: " + document.get("path"));
            System.out.println("Score: " + doc.score);

            for (int i = 0; i < 30; i++) {
                System.out.print("=");
            }
            System.out.println();
        }

        reader.close();
    }

}
