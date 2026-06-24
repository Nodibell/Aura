import SwiftUI
import UniformTypeIdentifiers

struct AnalysisSchedulerSheet: View {
    @Binding var isPresented: Bool
    
    // Optional current dataset context to pre-populate creation form
    var currentDatasetPath: String? = nil
    var currentTargetColumn: String? = nil
    var currentDatasetType: DatasetType = .tabular
    var currentConfig: AnalysisConfig? = nil
    
    @State private var tasks: [ScheduledTask] = []
    
    // New Task form state
    @State private var taskName: String = ""
    @State private var recurrenceType: RecurrenceType = .daily
    @State private var hourlyHours: Int = 1
    @State private var exportFormat: ExportFormat = .html
    @State private var destinationFolder: String = ""
    
    // UI tabs
    @State private var selectedTab = 0 // 0 = Active Schedules, 1 = Create Schedule
    
    enum RecurrenceType: String, CaseIterable, Identifiable {
        case hourly = "Hourly"
        case daily = "Daily"
        case weekly = "Weekly"
        
        var id: String { self.rawValue }
    }
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Analysis Scheduler")
                        .font(.title2)
                        .fontWeight(.bold)
                    Text("Automate analysis and export reports periodically")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                Spacer()
                
                Button(action: { isPresented = false }) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title3)
                        .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
            }
            .padding()
            .background(Color.primary.opacity(0.02))
            
            Divider()
            
            // Picker to toggle between views (only show if we have current dataset context)
            if currentDatasetPath != nil {
                Picker("", selection: $selectedTab) {
                    Text("Active Schedules").tag(0)
                    Text("Schedule Current Dataset").tag(1)
                }
                .pickerStyle(.segmented)
                .padding()
            }
            
            if selectedTab == 1 && currentDatasetPath != nil {
                createScheduleForm
            } else {
                schedulesList
            }
            
            Divider()
            
            // Footer
            HStack {
                Spacer()
                Button("Close") {
                    isPresented = false
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            }
            .padding()
            .background(Color(nsColor: .windowBackgroundColor))
        }
        .frame(width: 600, height: 500)
        
        .onAppear {
            refreshTasks()
            if currentDatasetPath != nil {
                // If opened with a current dataset, default to Create tab and pre-fill name
                selectedTab = 1
                let filename = URL(fileURLWithPath: currentDatasetPath!).deletingPathExtension().lastPathComponent
                taskName = "\(filename) Routine"
                
                // Try to set default export folder to Downloads
                if let downloadsURL = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first {
                    destinationFolder = downloadsURL.path
                }
            } else {
                selectedTab = 0
            }
        }
    }
    
    // MARK: - List View
    
    private var schedulesList: some View {
        VStack(spacing: 0) {
            if tasks.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "clock.badge.exclamationmark")
                        .font(.system(size: 40))
                        .foregroundColor(.secondary)
                    Text("No scheduled tasks configured")
                        .font(.headline)
                        .foregroundColor(.secondary)
                    Text("Schedule a recurring task from the summary view of any dataset.")
                        .font(.caption)
                        .foregroundColor(.secondary.opacity(0.8))
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 40)
                }
                .frame(maxHeight: .infinity)
            } else {
                List {
                    ForEach(tasks) { task in
                        HStack(alignment: .center, spacing: 14) {
                            // Icon and active status indicator
                            ZStack {
                                RoundedRectangle(cornerRadius: 8)
                                    .fill(task.isActive ? Color.purple.opacity(0.12) : Color.primary.opacity(0.04))
                                    .frame(width: 40, height: 40)
                                
                                Image(systemName: task.isActive ? "clock.fill" : "clock")
                                    .foregroundColor(task.isActive ? .purple : .secondary)
                                    .font(.title3)
                            }
                            
                            // Task details
                            VStack(alignment: .leading, spacing: 4) {
                                HStack(spacing: 6) {
                                    Text(task.name)
                                        .font(.headline)
                                        .foregroundColor(task.isActive ? .primary : .secondary)
                                    
                                    Text(task.recurrence.label)
                                        .font(.system(size: 9, weight: .bold))
                                        .foregroundColor(.white)
                                        .padding(.horizontal, 5)
                                        .padding(.vertical, 1)
                                        .background(Color.blue.opacity(task.isActive ? 0.8 : 0.4))
                                        .cornerRadius(4)
                                }
                                
                                Text(URL(fileURLWithPath: task.datasetPath).lastPathComponent)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                
                                HStack(spacing: 8) {
                                    Text("Next run: \(task.nextRun.formatted(date: .abbreviated, time: .shortened))")
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                    
                                    if let last = task.lastRun {
                                        Text("• Last run: \(last.formatted(date: .abbreviated, time: .shortened))")
                                            .font(.caption2)
                                            .foregroundColor(.secondary)
                                    }
                                }
                            }
                            
                            Spacer()
                            
                            // Actions Panel
                            HStack(spacing: 12) {
                                Button(action: {
                                    AnalysisScheduler.shared.triggerImmediately(task)
                                }) {
                                    Image(systemName: "play.fill")
                                        .help("Run immediately")
                                }
                                .buttonStyle(.plain)
                                .foregroundColor(.green)
                                
                                Toggle("", isOn: Binding(
                                    get: { task.isActive },
                                    set: { _ in toggleTask(task) }
                                ))
                                .toggleStyle(.switch)
                                .scaleEffect(0.75)
                                
                                Button(action: {
                                    deleteTask(task)
                                }) {
                                    Image(systemName: "trash")
                                        .foregroundColor(.red.opacity(0.8))
                                        .help("Delete task")
                                }
                                .buttonStyle(.plain)
                            }
                        }
                        .padding(.vertical, 8)
                    }
                }
                .listStyle(.sidebar)
            }
        }
        .frame(maxHeight: .infinity)
    }
    
    // MARK: - Creation Form
    
    private var createScheduleForm: some View {
        Form {
            Section(header: Text("Task Settings").fontWeight(.semibold)) {
                TextField("Task Name", text: $taskName)
                    .textFieldStyle(.roundedBorder)
                
                HStack {
                    Picker("Recurrence", selection: $recurrenceType) {
                        ForEach(RecurrenceType.allCases) { type in
                            Text(type.rawValue).tag(type)
                        }
                    }
                    .pickerStyle(.menu)
                    
                    if recurrenceType == .hourly {
                        Stepper(value: $hourlyHours, in: 1...24) {
                            Text("Every \(hourlyHours) hrs")
                        }
                        .frame(width: 140)
                    }
                }
            }
            
            Section(header: Text("Export Configuration").fontWeight(.semibold)) {
                Picker("Report Format", selection: $exportFormat) {
                    ForEach(ExportFormat.allCases) { format in
                        Text(format.rawValue).tag(format)
                    }
                }
                .pickerStyle(.inline)
                
                VStack(alignment: .leading, spacing: 6) {
                    Text("Destination Folder")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    HStack {
                        TextField("/path/to/export/reports", text: $destinationFolder)
                            .textFieldStyle(.roundedBorder)
                        
                        Button("Browse...") {
                            let panel = NSOpenPanel()
                            panel.allowsMultipleSelection = false
                            panel.canChooseDirectories = true
                            panel.canChooseFiles = false
                            if panel.runModal() == .OK, let url = panel.url {
                                destinationFolder = url.path
                            }
                        }
                    }
                }
            }
            
            Button(action: saveNewSchedule) {
                HStack {
                    Spacer()
                    Image(systemName: "calendar.badge.plus")
                    Text("Schedule Task")
                    Spacer()
                }
                .fontWeight(.bold)
                .foregroundColor(.white)
                .padding(.vertical, 10)
                .background(
                    taskName.trimmingCharacters(in: .whitespaces).isEmpty || destinationFolder.trimmingCharacters(in: .whitespaces).isEmpty
                    ? AnyShapeStyle(Color.gray.opacity(0.2))
                    : AnyShapeStyle(LinearGradient(colors: [.purple, .indigo], startPoint: .leading, endPoint: .trailing))
                )
                .cornerRadius(10)
            }
            .buttonStyle(.plain)
            .disabled(taskName.trimmingCharacters(in: .whitespaces).isEmpty || destinationFolder.trimmingCharacters(in: .whitespaces).isEmpty)
            .padding(.top, 16)
        }
        .padding()
        .frame(maxHeight: .infinity)
    }
    
    // MARK: - Operations
    
    private func refreshTasks() {
        tasks = AnalysisScheduler.shared.getTasks()
    }
    
    private func toggleTask(_ task: ScheduledTask) {
        AnalysisScheduler.shared.toggleTaskActive(withId: task.id)
        refreshTasks()
    }
    
    private func deleteTask(_ task: ScheduledTask) {
        AnalysisScheduler.shared.removeTask(withId: task.id)
        refreshTasks()
    }
    
    private func saveNewSchedule() {
        guard let datasetPath = currentDatasetPath else { return }
        
        let recurrence: Recurrence
        switch recurrenceType {
        case .hourly: recurrence = .hourly(hourlyHours)
        case .daily: recurrence = .daily
        case .weekly: recurrence = .weekly
        }
        
        let config = currentConfig ?? AnalysisConfig()
        
        let newTask = ScheduledTask(
            id: UUID(),
            name: taskName,
            datasetPath: datasetPath,
            targetColumn: currentTargetColumn,
            taskType: currentDatasetType,
            recurrence: recurrence,
            exportFormat: exportFormat,
            exportFolderPath: destinationFolder,
            isActive: true,
            lastRun: nil,
            nextRun: Date(), // First run immediately / as scheduled
            config: config
        )
        
        AnalysisScheduler.shared.addTask(newTask)
        selectedTab = 0
        refreshTasks()
        
        // Reset creation form
        taskName = ""
    }
}
