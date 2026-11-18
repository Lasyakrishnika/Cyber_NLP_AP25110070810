import javax.swing.*;
import javax.swing.border.TitledBorder;
import javax.swing.filechooser.FileNameExtensionFilter;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

// 1. Immutable Data Carriers using Java Records
record ThreatInput(String content, String origin, String customTimestamp) {
    ThreatInput(String content) {
        this(content, "Manual Input", null);
    }

    ThreatInput(String content, String origin) {
        this(content, origin, null);
    }
}

record AnalysisOutput(String predictedCategory, List<String> extractedKeywords, String origin, String analyzedTime) {
    static AnalysisOutput create(String category, List<String> keywords, String origin, String customTimestamp) {
        String formattedTime = (customTimestamp != null && !customTimestamp.isBlank())
                ? customTimestamp
                : LocalDateTime.now().format(DateTimeFormatter.ofPattern("dd-MM-yyyy HH:mm:ss"));
        return new AnalysisOutput(category, keywords, origin, formattedTime);
    }
}

// 2. Decoupled Text Normalization Utility
final class Tokenizer {
    private static final Set<String> IGNORED_TOKENS = Set.of(
        "the", "is", "a", "an", "and", "or", "of", "to",
        "in", "on", "for", "with", "by", "was", "were",
        "are", "be", "has", "have", "had", "this", "that",
        "from", "it", "as", "at", "their", "they", "into", "new", "using"
    );

    public List<String> tokenize(String rawText) {
        if (rawText == null || rawText.isBlank()) {
            return Collections.emptyList();
        }
        String normalized = rawText.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9\\s]", " ").trim();
        if (normalized.isEmpty()) {
            return Collections.emptyList();
        }
        return Arrays.stream(normalized.split("\\s+"))
                .filter(token -> !IGNORED_TOKENS.contains(token))
                .toList();
    }
}

// 3. Interface-Driven Classifier Engine (Composition over Inheritance)
interface ThreatClassifier {
    AnalysisOutput evaluate(ThreatInput input) throws IllegalArgumentException;
}

class ProbabilisticThreatModel implements ThreatClassifier {
    private final Tokenizer tokenizer;
    private final Map<String, Map<String, Long>> termFrequencyByClass = new LinkedHashMap<>();
    private final Map<String, Long> totalTermsByClass = new HashMap<>();
    private final Map<String, Integer> sampleCountByClass = new HashMap<>();
    private final Set<String> knownVocabulary = new HashSet<>();
    private int totalSamples = 0;

    public ProbabilisticThreatModel(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        seedTrainingCorpus();
    }

    private void seedTrainingCorpus() {
        Map<String, List<String>> corpus = Map.of(
            "Ransomware", List.of(
                "ransomware encrypt encrypted encryption files ransom payment victim locked restore decrypt extortion wannacry ryuk",
                "attackers encrypted documents demanded cryptocurrency bitcoin payment files inaccessible wiper destruction mbr"
            ),
            "Phishing", List.of(
                "phishing spearphishing spear email emails password credentials fraudulent malicious link fake login impersonation scam",
                "fraudulent message directed employees to fake website steal credentials lure decoy themed social engineering"
            ),
            "Malware", List.of(
                "malware trojan virus spyware infection infected malicious software payload executable backdoor rat rootkit worm downloader",
                "plugx winnti stuxnet duqu flame shamoon blackenergy turla lazarus hydraq aurora exploit zeroday vulnerability implant webshell"
            ),
            "Data Breach", List.of(
                "data breach stolen leaked database information exposed discloses records exfiltration confidential espionage cyberespionage",
                "apt operation campaign targeted attacks cyberattack cyberattacks warfare cyberwarfare hacking group intelligence surveillance reconnaissance diplomatic government secrets"
            ),
            "Denial of Service", List.of(
                "denial service ddos dos traffic server network overload requests unavailable flooding botnet disruption",
                "infrastructure outage command control c2 router disruption critical grid electric power"
            ),
            "Credential Theft", List.of(
                "credential credentials theft stolen username password authentication account login keylogger identity stealer infostealer",
                "harvests online credentials token session bypass mfa certificatehijacking"
            )
        );

        corpus.forEach((label, samples) -> samples.forEach(sample -> ingestSample(label, sample)));
    }

    private void ingestSample(String label, String sampleText) {
        List<String> tokens = tokenizer.tokenize(sampleText);
        termFrequencyByClass.computeIfAbsent(label, k -> new HashMap<>());
        sampleCountByClass.merge(label, 1, Integer::sum);
        totalSamples++;

        Map<String, Long> classMap = termFrequencyByClass.get(label);
        Map<String, Long> batchCounts = tokens.stream()
                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));

        batchCounts.forEach((term, freq) -> {
            classMap.merge(term, freq, Long::sum);
            totalTermsByClass.merge(label, freq, Long::sum);
            knownVocabulary.add(term);
        });
    }

    @Override
    public AnalysisOutput evaluate(ThreatInput input) throws IllegalArgumentException {
        if (input == null) {
            throw new IllegalArgumentException("Report cannot be null.");
        }
        List<String> tokens = tokenizer.tokenize(input.content());
        if (tokens.isEmpty()) {
            throw new IllegalArgumentException("Please enter a valid cyber threat report.");
        }

        String topCategory = "Unknown";
        double maxLogProb = Double.NEGATIVE_INFINITY;
        int vocabSize = knownVocabulary.size();

        for (String category : termFrequencyByClass.keySet()) {
            double logProb = Math.log((double) sampleCountByClass.get(category) / totalSamples);
            Map<String, Long> termMap = termFrequencyByClass.get(category);
            long classTotalWords = totalTermsByClass.get(category);

            for (String token : tokens) {
                long occurrences = termMap.getOrDefault(token, 0L);
                logProb += Math.log((occurrences + 1.0) / (classTotalWords + vocabSize));
            }

            if (logProb > maxLogProb) {
                maxLogProb = logProb;
                topCategory = category;
            }
        }

        List<String> matchedKeywords = tokens.stream()
                .filter(knownVocabulary::contains)
                .distinct()
                .limit(8)
                .toList();

        if (matchedKeywords.isEmpty()) {
            matchedKeywords = List.of("No known keywords found");
        }

        return AnalysisOutput.create(topCategory, matchedKeywords, input.origin(), input.customTimestamp());
    }
}

// 4. Dedicated File, Dataset & CSV Persistence Handler
final class FileStorageService {
    public static String readTextFile(Path filePath) throws IOException {
        return Files.readString(filePath, StandardCharsets.UTF_8);
    }

    public static void writeTableToCsv(Path targetPath, DefaultTableModel tableModel) throws IOException {
        List<String> lines = new ArrayList<>();
        lines.add("Date and Time,Source,Category,Keywords");

        for (int row = 0; row < tableModel.getRowCount(); row++) {
            String line = String.join(",",
                escapeCsv(tableModel.getValueAt(row, 0)),
                escapeCsv(tableModel.getValueAt(row, 1)),
                escapeCsv(tableModel.getValueAt(row, 2)),
                escapeCsv(tableModel.getValueAt(row, 3))
            );
            lines.add(line);
        }
        Files.write(targetPath, lines, StandardCharsets.UTF_8);
    }

    // Parses either an exported classification_history.csv OR an APTnotes CSV/JSON dataset
    public static List<String[]> loadHistoryOrDataset(Path filePath, ThreatClassifier classifier) throws IOException {
        String fileName = filePath.getFileName().toString().toLowerCase(Locale.ROOT);
        if (fileName.endsWith(".json")) {
            return parseJsonDataset(filePath, classifier);
        }

        List<String> lines = Files.readAllLines(filePath, StandardCharsets.UTF_8);
        List<String[]> rows = new ArrayList<>();
        if (lines.isEmpty()) {
            return rows;
        }

        String header = lines.get(0).toLowerCase(Locale.ROOT);
        boolean isExportedHistory = header.contains("date and time") && header.contains("category");

        for (int i = 1; i < lines.size(); i++) {
            String line = lines.get(i).trim();
            if (line.isEmpty()) continue;
            List<String> cols = parseCsvLine(line);

            if (isExportedHistory && cols.size() >= 4) {
                rows.add(new String[]{cols.get(0), cols.get(1), cols.get(2), cols.get(3), cols.get(1)});
            } else if (cols.size() >= 6) {
                // APTnotes CSV format: Filename,Title,Source,Link,SHA-1,Date,Year
                String docFile = cols.get(0);
                String title = cols.get(1);
                String vendor = cols.get(2);
                String date = cols.get(5);
                String fullReportText = "Report Title: " + title + "\nSource Vendor: " + vendor + "\nFile: " + docFile + "\nDate: " + date;
                try {
                    AnalysisOutput out = classifier.evaluate(new ThreatInput(title + " " + docFile, vendor + " (" + docFile + ")", date));
                    rows.add(new String[]{out.analyzedTime(), out.origin(), out.predictedCategory(), String.join(", ", out.extractedKeywords()), fullReportText});
                } catch (Exception ignored) {
                }
            }
        }
        return rows;
    }

    private static List<String[]> parseJsonDataset(Path filePath, ThreatClassifier classifier) throws IOException {
        String content = Files.readString(filePath, StandardCharsets.UTF_8);
        List<String[]> rows = new ArrayList<>();
        Pattern objPattern = Pattern.compile("\\{([^}]+)\\}");
        Matcher matcher = objPattern.matcher(content);

        while (matcher.find()) {
            String block = matcher.group(1);
            String filename = extractJsonField(block, "Filename");
            String title = extractJsonField(block, "Title");
            String source = extractJsonField(block, "Source");
            String date = extractJsonField(block, "Date");

            if (!title.isEmpty()) {
                String fullReportText = "Report Title: " + title + "\nSource Vendor: " + source + "\nFile: " + filename + "\nDate: " + date;
                try {
                    AnalysisOutput out = classifier.evaluate(new ThreatInput(title + " " + filename, source + " (" + filename + ")", date));
                    rows.add(new String[]{out.analyzedTime(), out.origin(), out.predictedCategory(), String.join(", ", out.extractedKeywords()), fullReportText});
                } catch (Exception ignored) {
                }
            }
        }
        return rows;
    }

    private static String extractJsonField(String block, String key) {
        Pattern p = Pattern.compile("\"" + key + "\"\\s*:\\s*\"((?:\\\\\"|[^\"])*)\"");
        Matcher m = p.matcher(block);
        return m.find() ? m.group(1).replace("\\\"", "\"") : "";
    }

    public static List<String> parseCsvLine(String line) {
        List<String> values = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean inQuotes = false;

        for (int i = 0; i < line.length(); i++) {
            char ch = line.charAt(i);
            if (ch == '"') {
                if (inQuotes && i + 1 < line.length() && line.charAt(i + 1) == '"') {
                    current.append('"');
                    i++;
                } else {
                    inQuotes = !inQuotes;
                }
            } else if (ch == ',' && !inQuotes) {
                values.add(current.toString().trim());
                current.setLength(0);
            } else {
                current.append(ch);
            }
        }
        values.add(current.toString().trim());
        return values;
    }

    private static String escapeCsv(Object cellValue) {
        return "\"" + String.valueOf(cellValue).replace("\"", "\"\"") + "\"";
    }
}

// 5. Modular Swing User Interface
class ThreatIntelWorkspace extends JFrame {
    private static final Color BG_DARK = new Color(19, 27, 43);
    private static final Color SURFACE_SLATE = new Color(30, 41, 59);
    private static final Color ACCENT_CYAN = new Color(56, 189, 248);
    private static final Color TEXT_LIGHT = new Color(235, 241, 250);

    private final ThreatClassifier classifier;
    private final JTextArea inputBox;
    private final JTextArea outputBox;
    private final JLabel footerStatus;
    private final JButton runBtn;
    private final JTabbedPane viewTabs;
    private final DefaultTableModel logTableModel;
    private String activeSource = "Manual Input";

    public ThreatIntelWorkspace(ThreatClassifier classifier) {
        this.classifier = classifier;

        setTitle("CyberThreat Intel | Threat Report Classifier");
        setSize(1080, 740);
        setMinimumSize(new Dimension(880, 620));
        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);

        JPanel mainContainer = new JPanel(new BorderLayout(12, 12));
        mainContainer.setBackground(BG_DARK);
        mainContainer.setBorder(BorderFactory.createEmptyBorder(18, 18, 18, 18));

        mainContainer.add(buildBanner(), BorderLayout.NORTH);

        inputBox = createEditorArea(10, true);
        outputBox = createEditorArea(5, false);

        JScrollPane inputWrapper = wrapWithTitledScroll(
            inputBox,
            "Threat Report Input (Type/paste a report below, or click 'Import Report File')"
        );
        JScrollPane outputWrapper = wrapWithTitledScroll(outputBox, "Analysis Results");

        // Clear, self-explanatory button labels and tooltips
        JButton openFileBtn = new JButton("Import Report File (.txt / .csv / .json)");
        runBtn = new JButton("Analyze Input Text");
        JButton resetBtn = new JButton("Clear Input");
        JButton loadHistoryBtn = new JButton("Load History (.csv / .json)");
        JButton saveCsvBtn = new JButton("Export History (.csv)");

        openFileBtn.setToolTipText("Load a .txt report into the editor, or pick a report from APTnotes (.csv / .json)");
        runBtn.setToolTipText("Classify the threat report currently shown in the input box");
        resetBtn.setToolTipText("Clear the input box and analysis results");
        loadHistoryBtn.setToolTipText("Import saved history (.csv) or batch-load APTnotes (.csv / .json) into the History tab");
        saveCsvBtn.setToolTipText("Save all rows in the Session History table to a CSV file");

        JPanel actionToolbar = new JPanel(new FlowLayout(FlowLayout.CENTER, 10, 8));
        actionToolbar.setBackground(BG_DARK);
        actionToolbar.add(openFileBtn);
        actionToolbar.add(runBtn);
        actionToolbar.add(resetBtn);
        actionToolbar.add(loadHistoryBtn);
        actionToolbar.add(saveCsvBtn);

        JLabel helperHint = new JLabel(
            "Tip: Use 'Import Report File' to open a .txt report or pick from APTnotes, or use 'Load History' to populate the History tab.",
            SwingConstants.CENTER
        );
        helperHint.setForeground(new Color(170, 190, 215));
        helperHint.setFont(new Font("Arial", Font.ITALIC, 12));

        JPanel controlsSection = new JPanel(new BorderLayout(4, 4));
        controlsSection.setBackground(BG_DARK);
        controlsSection.add(helperHint, BorderLayout.NORTH);
        controlsSection.add(actionToolbar, BorderLayout.CENTER);

        JPanel analyzerTab = new JPanel(new BorderLayout(10, 10));
        analyzerTab.setBackground(BG_DARK);
        analyzerTab.add(inputWrapper, BorderLayout.CENTER);
        analyzerTab.add(controlsSection, BorderLayout.SOUTH);

        logTableModel = new DefaultTableModel(new String[]{"Date and Time", "Source", "Category", "Keywords"}, 0) {
            @Override
            public boolean isCellEditable(int r, int c) {
                return false;
            }
        };

        JTable logTable = new JTable(logTableModel);
        logTable.setRowHeight(24);
        logTable.setAutoResizeMode(JTable.AUTO_RESIZE_ALL_COLUMNS);
        logTable.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);

        // Selecting a row in History shows its details in Analysis Results
        logTable.getSelectionModel().addListSelectionListener(e -> {
            if (!e.getValueIsAdjusting() && logTable.getSelectedRow() != -1) {
                int row = logTable.getSelectedRow();
                outputBox.setText(
                    "Predicted Threat Category: " + logTableModel.getValueAt(row, 2) +
                    "\n\nImportant Keywords: " + logTableModel.getValueAt(row, 3) +
                    "\n\nSource: " + logTableModel.getValueAt(row, 1) +
                    "\nAnalyzed At: " + logTableModel.getValueAt(row, 0) +
                    "\n\nNote: Selected from Session History."
                );
            }
        });

        JScrollPane historyScroll = wrapWithTitledScroll(
            logTable,
            "Session History (Click any row to inspect its result below)"
        );

        JButton historyLoadBtn = new JButton("Load History from File (.csv / .json)");
        JButton historyExportBtn = new JButton("Export History to CSV");
        JButton historyClearBtn = new JButton("Clear History Table");

        JPanel historyToolbar = new JPanel(new FlowLayout(FlowLayout.CENTER, 12, 8));
        historyToolbar.setBackground(BG_DARK);
        historyToolbar.add(historyLoadBtn);
        historyToolbar.add(historyExportBtn);
        historyToolbar.add(historyClearBtn);

        JPanel historyTab = new JPanel(new BorderLayout(10, 10));
        historyTab.setBackground(BG_DARK);
        historyTab.add(historyScroll, BorderLayout.CENTER);
        historyTab.add(historyToolbar, BorderLayout.SOUTH);

        viewTabs = new JTabbedPane();
        viewTabs.addTab("Analyze Report", analyzerTab);
        viewTabs.addTab("Session History (0)", historyTab);

        footerStatus = new JLabel("Ready — Type a report or click 'Import Report File (.txt / .csv / .json)'");
        footerStatus.setForeground(ACCENT_CYAN);

        JPanel bottomPane = new JPanel(new BorderLayout(10, 10));
        bottomPane.setBackground(BG_DARK);
        bottomPane.add(outputWrapper, BorderLayout.CENTER);
        bottomPane.add(footerStatus, BorderLayout.SOUTH);

        JPanel bodyLayout = new JPanel(new BorderLayout(10, 10));
        bodyLayout.setBackground(BG_DARK);
        bodyLayout.add(viewTabs, BorderLayout.CENTER);
        bodyLayout.add(bottomPane, BorderLayout.SOUTH);

        mainContainer.add(bodyLayout, BorderLayout.CENTER);
        setContentPane(mainContainer);

        // Event Bindings
        openFileBtn.addActionListener(e -> handleFileUpload());
        runBtn.addActionListener(e -> executeClassification());
        resetBtn.addActionListener(e -> resetWorkspace());
        loadHistoryBtn.addActionListener(e -> handleHistoryImport());
        saveCsvBtn.addActionListener(e -> handleCsvExport());

        historyLoadBtn.addActionListener(e -> handleHistoryImport());
        historyExportBtn.addActionListener(e -> handleCsvExport());
        historyClearBtn.addActionListener(e -> {
            logTableModel.setRowCount(0);
            updateHistoryTabTitle();
            footerStatus.setText("Session history cleared.");
        });

        // Auto-load existing classification_history.csv if present in project directory
        preloadExistingHistory();
    }

    private void updateHistoryTabTitle() {
        viewTabs.setTitleAt(1, "Session History (" + logTableModel.getRowCount() + ")");
    }

    private void preloadExistingHistory() {
        Path defaultHistory = Path.of("classification_history.csv");
        if (Files.exists(defaultHistory)) {
            try {
                List<String[]> rows = FileStorageService.loadHistoryOrDataset(defaultHistory, classifier);
                for (String[] r : rows) {
                    logTableModel.addRow(new Object[]{r[0], r[1], r[2], r[3]});
                }
                updateHistoryTabTitle();
                if (!rows.isEmpty()) {
                    footerStatus.setText("Ready — Restored " + rows.size() + " previous records into Session History.");
                }
            } catch (IOException ignored) {
            }
        }
    }

    private JPanel buildBanner() {
        JLabel title = new JLabel("CyberThreat Intel", SwingConstants.CENTER);
        title.setForeground(ACCENT_CYAN);
        title.setFont(new Font("Arial", Font.BOLD, 28));

        JLabel desc = new JLabel("NLP-Based Classification of Cyber Threat Intelligence Reports", SwingConstants.CENTER);
        desc.setForeground(TEXT_LIGHT);
        desc.setFont(new Font("Arial", Font.PLAIN, 13));

        JPanel banner = new JPanel(new GridLayout(2, 1, 4, 4));
        banner.setBackground(BG_DARK);
        banner.add(title);
        banner.add(desc);
        return banner;
    }

    private JTextArea createEditorArea(int rows, boolean editable) {
        JTextArea area = new JTextArea(rows, 50);
        area.setEditable(editable);
        area.setLineWrap(true);
        area.setWrapStyleWord(true);
        area.setFont(new Font("Consolas", Font.PLAIN, 14));
        area.setBackground(SURFACE_SLATE);
        area.setForeground(TEXT_LIGHT);
        if (editable) {
            area.setCaretColor(ACCENT_CYAN);
        }
        area.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        return area;
    }

    private JScrollPane wrapWithTitledScroll(Component view, String title) {
        JScrollPane pane = new JScrollPane(view);
        TitledBorder border = BorderFactory.createTitledBorder(title);
        pane.setBorder(border);
        return pane;
    }

    // Improved Load Report: Starts in current project folder and supports .txt, .csv, .xls, and .json
    private void handleFileUpload() {
        JFileChooser filePicker = new JFileChooser(new File("."));
        filePicker.setDialogTitle("Select a Threat Report (.txt) or Dataset (.csv / .json)");
        filePicker.setFileFilter(new FileNameExtensionFilter(
            "Threat Reports & Datasets (*.txt, *.csv, *.xls, *.json)", "txt", "csv", "xls", "json"
        ));

        if (filePicker.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) {
            return;
        }

        Path selected = filePicker.getSelectedFile().toPath();
        String fileName = selected.getFileName().toString();
        String lower = fileName.toLowerCase(Locale.ROOT);

        try {
            if (lower.endsWith(".txt")) {
                inputBox.setText(FileStorageService.readTextFile(selected));
                activeSource = fileName;
                viewTabs.setSelectedIndex(0);
                footerStatus.setText("Loaded '" + fileName + "' — Analyzing report...");
                executeClassification();
            } else if (lower.endsWith(".csv") || lower.endsWith(".xls") || lower.endsWith(".json")) {
                List<String[]> entries = FileStorageService.loadHistoryOrDataset(selected, classifier);
                if (entries.isEmpty()) {
                    JOptionPane.showMessageDialog(this, "No valid report entries found in " + fileName, "Empty Dataset", JOptionPane.WARNING_MESSAGE);
                    return;
                }
                promptDatasetAction(fileName, entries);
            } else {
                JOptionPane.showMessageDialog(
                    this,
                    "Please select a .txt, .csv, .xls, or .json file.",
                    "Unsupported File Format",
                    JOptionPane.WARNING_MESSAGE
                );
            }
        } catch (IOException ex) {
            JOptionPane.showMessageDialog(this, "Unable to read file: " + ex.getMessage(), "File Error", JOptionPane.ERROR_MESSAGE);
        }
    }

    // Gives the user a convenient choice when selecting a multi-report CSV/JSON file
    private void promptDatasetAction(String fileName, List<String[]> entries) {
        String[] options = {
            "Pick 1 Report to Analyze",
            "Load All (" + entries.size() + ") into History",
            "Cancel"
        };

        int choice = JOptionPane.showOptionDialog(
            this,
            "File '" + fileName + "' contains " + entries.size() + " threat reports.\nHow would you like to use it?",
            "Dataset Loaded",
            JOptionPane.YES_NO_CANCEL_OPTION,
            JOptionPane.QUESTION_MESSAGE,
            null,
            options,
            options[0]
        );

        if (choice == 0) {
            String[] reportTitles = entries.stream()
                    .map(r -> r[0] + " | " + r[1] + " -> [" + r[2] + "]")
                    .toArray(String[]::new);

            JList<String> list = new JList<>(reportTitles);
            list.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
            list.setSelectedIndex(0);
            JScrollPane scrollPane = new JScrollPane(list);
            scrollPane.setPreferredSize(new Dimension(650, 320));

            int picked = JOptionPane.showConfirmDialog(
                this,
                scrollPane,
                "Select a Threat Report from " + fileName,
                JOptionPane.OK_CANCEL_OPTION,
                JOptionPane.PLAIN_MESSAGE
            );

            if (picked == JOptionPane.OK_OPTION && list.getSelectedIndex() != -1) {
                String[] chosen = entries.get(list.getSelectedIndex());
                inputBox.setText(chosen[4]);
                activeSource = chosen[1];
                viewTabs.setSelectedIndex(0);
                executeClassification();
            }
        } else if (choice == 1) {
            for (String[] r : entries) {
                logTableModel.addRow(new Object[]{r[0], r[1], r[2], r[3]});
            }
            updateHistoryTabTitle();
            viewTabs.setSelectedIndex(1);
            footerStatus.setText("Loaded " + entries.size() + " reports from '" + fileName + "' into Session History.");
        }
    }

    // Dedicated Load History handler
    private void handleHistoryImport() {
        JFileChooser filePicker = new JFileChooser(new File("."));
        filePicker.setDialogTitle("Select History CSV or APTnotes Dataset (.csv / .xls / .json)");
        filePicker.setFileFilter(new FileNameExtensionFilter(
            "History & Dataset Files (*.csv, *.xls, *.json)", "csv", "xls", "json"
        ));

        if (filePicker.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) {
            return;
        }

        Path selected = filePicker.getSelectedFile().toPath();
        try {
            List<String[]> rows = FileStorageService.loadHistoryOrDataset(selected, classifier);
            if (rows.isEmpty()) {
                JOptionPane.showMessageDialog(this, "No history records found in selected file.", "Import History", JOptionPane.WARNING_MESSAGE);
                return;
            }

            for (String[] r : rows) {
                logTableModel.addRow(new Object[]{r[0], r[1], r[2], r[3]});
            }
            updateHistoryTabTitle();
            viewTabs.setSelectedIndex(1);
            footerStatus.setText("Loaded " + rows.size() + " records from '" + selected.getFileName() + "' into Session History.");
            JOptionPane.showMessageDialog(
                this,
                "Successfully loaded " + rows.size() + " records into Session History.",
                "History Loaded",
                JOptionPane.INFORMATION_MESSAGE
            );
        } catch (IOException ex) {
            JOptionPane.showMessageDialog(this, "Unable to load history: " + ex.getMessage(), "File Error", JOptionPane.ERROR_MESSAGE);
        }
    }

    private void executeClassification() {
        String rawInput = inputBox.getText();
        String sourceName = activeSource;

        runBtn.setEnabled(false);
        outputBox.setText("Analyzing report...");
        footerStatus.setText("Processing in background...");

        new SwingWorker<AnalysisOutput, Void>() {
            @Override
            protected AnalysisOutput doInBackground() {
                return classifier.evaluate(new ThreatInput(rawInput, sourceName));
            }

            @Override
            protected void done() {
                try {
                    AnalysisOutput res = get();
                    String keywordsJoined = String.join(", ", res.extractedKeywords());

                    outputBox.setText(
                        "Predicted Threat Category: " + res.predictedCategory() +
                        "\n\nImportant Keywords: " + keywordsJoined +
                        "\n\nSource: " + res.origin() +
                        "\nAnalyzed At: " + res.analyzedTime() +
                        "\n\nNote: Prototype prediction; not a validated security assessment."
                    );

                    logTableModel.addRow(new Object[]{
                        res.analyzedTime(),
                        res.origin(),
                        res.predictedCategory(),
                        keywordsJoined
                    });
                    updateHistoryTabTitle();

                    footerStatus.setText("Classification completed — Added to Session History.");
                } catch (InterruptedException ex) {
                    Thread.currentThread().interrupt();
                    footerStatus.setText("Analysis interrupted.");
                } catch (ExecutionException ex) {
                    Throwable rootCause = ex.getCause();
                    outputBox.setText(rootCause != null ? rootCause.getMessage() : "Unable to analyze the report.");
                    footerStatus.setText("Classification failed.");
                } finally {
                    runBtn.setEnabled(true);
                }
            }
        }.execute();
    }

    private void resetWorkspace() {
        inputBox.setText("");
        outputBox.setText("");
        activeSource = "Manual Input";
        footerStatus.setText("Ready — Type a report or click 'Import Report File (.txt / .csv / .json)'");
    }

    private void handleCsvExport() {
        JFileChooser savePicker = new JFileChooser(new File("."));
        savePicker.setDialogTitle("Save Session History to CSV");
        savePicker.setSelectedFile(new File("classification_history.csv"));
        if (savePicker.showSaveDialog(this) != JFileChooser.APPROVE_OPTION) {
            return;
        }
        try {
            FileStorageService.writeTableToCsv(savePicker.getSelectedFile().toPath(), logTableModel);
            footerStatus.setText("Exported " + logTableModel.getRowCount() + " records to '" + savePicker.getSelectedFile().getName() + "'.");
            JOptionPane.showMessageDialog(this, "History exported successfully.");
        } catch (IOException ex) {
            JOptionPane.showMessageDialog(this, "Unable to export history: " + ex.getMessage(), "File Error", JOptionPane.ERROR_MESSAGE);
        }
    }
}

public class Main {

    public Main() {
        applySystemTheme();
        ThreatClassifier classifier = new ProbabilisticThreatModel(new Tokenizer());
        ThreatIntelWorkspace appWindow = new ThreatIntelWorkspace(classifier);
        appWindow.setVisible(true);
    }

    private void applySystemTheme() {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception ex) {
            System.err.println("Using default interface appearance.");
        }
    }

    public static void main(String[] args) {
        EventQueue.invokeLater(Main::new);
    }
}
