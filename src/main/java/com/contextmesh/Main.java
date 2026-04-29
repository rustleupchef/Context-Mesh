package com.contextmesh;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
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

        for (File file : contextFiles) {
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

            // Lines Segmentation
            // Paragraphs Segmentation
            // Pages Segmentation
            // Sentences Segmentation
            // Sections Segmentation
            // Key Information Extraction
        }

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
    
    
            for (Paired pair : pairs) {
                float[] vector = embedder.predict(pair.text);
    
                Document doc = new Document();
                doc.add(new KnnFloatVectorField("embedding", vector, VectorSimilarityFunction.COSINE));
                doc.add(new TextField("content", pair.text, Field.Store.YES));
                doc.add(new TextField("path", pair.path, Field.Store.YES));
                doc.add(new TextField("basePath", pair.basePath, Field.Store.YES));
                writer.addDocument(doc);
            }
            writer.commit();
            writer.close();
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
