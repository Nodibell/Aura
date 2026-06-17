import SwiftUI

// MARK: - Full Table View (Data Tab in Analysis Results)

struct FullTableView: View {
    let preview: FullTablePreview

    @State private var searchText: String = ""
    @State private var sortColumn: String? = nil
    @State private var sortAscending: Bool = true
    @State private var displayedCount: Int = 100

    private let colWidth: CGFloat = 148
    private let pageSize: Int = 100

    // Filtered + sorted rows
    private var filteredRows: [[String]] {
        var rows = preview.rows
        if !searchText.isEmpty {
            let q = searchText.lowercased()
            rows = rows.filter { row in row.contains { $0.lowercased().contains(q) } }
        }
        if let col = sortColumn, let idx = preview.columns.firstIndex(of: col) {
            rows.sort { a, b in
                let av = idx < a.count ? a[idx] : ""
                let bv = idx < b.count ? b[idx] : ""
                // Try numeric sort first
                if let an = Double(av), let bn = Double(bv) {
                    return sortAscending ? an < bn : an > bn
                }
                return sortAscending ? av < bv : av > bv
            }
        }
        return rows
    }

    var body: some View {
        VStack(spacing: 0) {
            // ── Toolbar ──────────────────────────────────────────────────────
            toolbar
            Divider().background(Color.white.opacity(0.06))

            // ── Table (Sticky Header Layout) ─────────────────────────────────
            GeometryReader { geometry in
                ScrollView(.horizontal, showsIndicators: true) {
                    VStack(alignment: .leading, spacing: 0) {
                        // Header
                        HStack(spacing: 0) {
                            // Row number column
                            Text("#")
                                .font(.system(size: 10, weight: .bold, design: .rounded))
                                .foregroundColor(.secondary.opacity(0.5))
                                .frame(width: 42, height: 36, alignment: .center)
                                .background(Color.white.opacity(0.04))
                                .border(Color.white.opacity(0.06), width: 0.5)

                            ForEach(preview.columns, id: \.self) { col in
                                SortableHeaderCell(
                                    title: col,
                                    isSorted: sortColumn == col,
                                    ascending: sortAscending
                                ) {
                                    if sortColumn == col {
                                        sortAscending.toggle()
                                    } else {
                                        sortColumn = col
                                        sortAscending = true
                                    }
                                }
                            }
                        }
                        .padding(.horizontal, 20)

                        ScrollView(.vertical, showsIndicators: true) {
                            VStack(alignment: .leading, spacing: 0) {
                                // Data rows (virtualized slice)
                                let visible = Array(filteredRows.prefix(displayedCount))
                                ForEach(Array(visible.enumerated()), id: \.offset) { idx, row in
                                    HStack(spacing: 0) {
                                        // Row number
                                        Text("\(idx + 1)")
                                            .font(.system(size: 10, design: .monospaced))
                                            .foregroundColor(.secondary.opacity(0.4))
                                            .frame(width: 42, height: 32, alignment: .center)
                                            .background(idx % 2 == 0 ? Color.white.opacity(0.015) : Color.clear)
                                            .border(Color.white.opacity(0.04), width: 0.5)

                                        ForEach(0..<preview.columns.count, id: \.self) { colIdx in
                                            let val = colIdx < row.count ? row[colIdx] : ""
                                            DataCellView(value: val, rowIndex: idx)
                                        }
                                    }
                                }

                                // Load more / footer
                                if filteredRows.count > displayedCount {
                                    Button {
                                        displayedCount = min(displayedCount + pageSize, filteredRows.count)
                                    } label: {
                                        HStack(spacing: 6) {
                                            Image(systemName: "arrow.down.circle")
                                            Text("Load \(min(pageSize, filteredRows.count - displayedCount)) more rows")
                                                .fontWeight(.semibold)
                                        }
                                        .font(.system(size: 12))
                                        .foregroundColor(.purple)
                                        .frame(width: 42 + CGFloat(preview.columns.count) * 148, height: 36)
                                        .background(Color.purple.opacity(0.06))
                                        .cornerRadius(8)
                                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.purple.opacity(0.2), lineWidth: 1))
                                    }
                                    .buttonStyle(.plain)
                                    .padding(.vertical, 16)
                                }
                            }
                        }
                        .frame(height: max(0, geometry.size.height - 36 - 16))
                        .padding(.horizontal, 20)
                    }
                }
            }
        }
    }

    // MARK: - Toolbar

    private var toolbar: some View {
        HStack(spacing: 12) {
            // Search
            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass")
                    .font(.system(size: 12))
                    .foregroundColor(.secondary)
                TextField("Search values…", text: $searchText)
                    .textFieldStyle(.plain)
                    .font(.system(size: 12))
                    .onChange(of: searchText) { _, _ in
                        displayedCount = pageSize   // reset pagination on new search
                    }
                if !searchText.isEmpty {
                    Button { searchText = "" } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.secondary)
                            .font(.system(size: 11))
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(Color.white.opacity(0.04))
            .cornerRadius(8)
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.07), lineWidth: 1))
            .frame(maxWidth: 260)

            Spacer()

            // Row count badge
            let showing = min(displayedCount, filteredRows.count)
            Text("Showing \(showing) of \(filteredRows.count) rows\(preview.totalRows > preview.rows.count ? " (full dataset: \(preview.totalRows))" : "")")
                .font(.system(size: 11))
                .foregroundColor(.secondary)

            // Reset sort
            if sortColumn != nil {
                Button {
                    sortColumn = nil
                    sortAscending = true
                } label: {
                    Label("Clear Sort", systemImage: "xmark.circle")
                        .font(.system(size: 11))
                        .foregroundColor(.orange)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
    }
}

// MARK: - Sortable Header Cell

private struct SortableHeaderCell: View {
    let title: String
    let isSorted: Bool
    let ascending: Bool
    let width: CGFloat = 148
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 4) {
                Text(title)
                    .font(.system(size: 11, weight: .bold, design: .rounded))
                    .foregroundColor(isSorted ? .purple : .primary)
                    .lineLimit(1)
                if isSorted {
                    Image(systemName: ascending ? "chevron.up" : "chevron.down")
                        .font(.system(size: 8, weight: .bold))
                        .foregroundColor(.purple)
                }
            }
            .padding(.horizontal, 10)
            .frame(width: width, height: 36, alignment: .leading)
            .background(isSorted ? Color.purple.opacity(0.08) : Color.white.opacity(0.04))
            .border(Color.white.opacity(0.06), width: 0.5)
        }
        .buttonStyle(.plain)
        .help("Sort by \(title)")
    }
}

// MARK: - Data Cell

private struct DataCellView: View {
    let value: String
    let rowIndex: Int
    let width: CGFloat = 148

    var body: some View {
        Text(value.isEmpty ? "—" : value)
            .font(.system(size: 11, design: .monospaced))
            .foregroundColor(cellColor)
            .lineLimit(1)
            .truncationMode(.middle)
            .padding(.horizontal, 10)
            .frame(width: width, height: 32, alignment: .leading)
            .background(rowIndex % 2 == 0 ? Color.white.opacity(0.015) : Color.clear)
            .border(Color.white.opacity(0.04), width: 0.5)
    }

    private var cellColor: Color {
        if value.isEmpty { return .gray.opacity(0.35) }
        if Double(value) != nil { return Color(hue: 0.6, saturation: 0.6, brightness: 0.85) }
        if value.lowercased() == "true" || value.lowercased() == "false" { return .purple }
        return .secondary
    }
}
