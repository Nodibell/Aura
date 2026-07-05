import SwiftUI

struct OnboardingView: View {
    @Binding var isPresented: Bool
    @State private var currentPage = 0
    @State private var animateAurora = false
    
    private let totalPages = 4
    
    var body: some View {
        ZStack {
            // ── Background Aurora Glow ──────────────────────────────────────
            ZStack {
                Color(nsColor: .windowBackgroundColor)
                
                // Glowing shapes for premium aesthetic
                Circle()
                    .fill(LinearGradient(colors: [.purple.opacity(0.18), .indigo.opacity(0.08)], startPoint: .topLeading, endPoint: .bottomTrailing))
                    .frame(width: 380, height: 380)
                    .blur(radius: 80)
                    .offset(x: animateAurora ? -120 : -80, y: animateAurora ? -100 : -60)
                
                Circle()
                    .fill(LinearGradient(colors: [.blue.opacity(0.12), .cyan.opacity(0.04)], startPoint: .topLeading, endPoint: .bottomTrailing))
                    .frame(width: 420, height: 420)
                    .blur(radius: 90)
                    .offset(x: animateAurora ? 140 : 100, y: animateAurora ? 120 : 80)
            }
            .ignoresSafeArea()
            .onAppear {
                withAnimation(.easeInOut(duration: 6).repeatForever(autoreverses: true)) {
                    animateAurora = true
                }
            }
            
            // ── Main Page Content ───────────────────────────────────────────
            VStack(spacing: 0) {
                // Top close button
                HStack {
                    Spacer()
                    Button {
                        isPresented = false
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title3)
                            .foregroundColor(.secondary.opacity(0.5))
                    }
                    .buttonStyle(.plain)
                    .padding([.top, .trailing], 16)
                }
                
                ZStack {
                    if currentPage == 0 {
                        welcomeSlide
                            .transition(.asymmetric(insertion: .move(edge: .trailing), removal: .move(edge: .leading)))
                    } else if currentPage == 1 {
                        featuresSlide
                            .transition(.asymmetric(insertion: .move(edge: .trailing), removal: .move(edge: .leading)))
                    } else if currentPage == 2 {
                        timeSeriesSlide
                            .transition(.asymmetric(insertion: .move(edge: .trailing), removal: .move(edge: .leading)))
                    } else if currentPage == 3 {
                        aiSlide
                            .transition(.asymmetric(insertion: .move(edge: .trailing), removal: .move(edge: .leading)))
                    }
                }
                .frame(maxHeight: .infinity)
                
                // Bottom controls panel
                bottomPanel
            }
        }
        .frame(width: 580, height: 480)
        .preferredColorScheme(.dark)
    }
    
    // MARK: - Slides
    
    private var welcomeSlide: some View {
        VStack(spacing: 24) {
            ZStack {
                // Glow behind logo
                Circle()
                    .fill(LinearGradient(colors: [.purple, .indigo], startPoint: .topLeading, endPoint: .bottomTrailing))
                    .frame(width: 110, height: 110)
                    .blur(radius: 18)
                    .opacity(0.3)
                
                // Icon Stack
                Image(systemName: "chart.bar.doc.horizontal.fill")
                    .font(.system(size: 60, weight: .bold))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [.purple, .indigo, .blue],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .shadow(color: .purple.opacity(0.4), radius: 12, x: 0, y: 6)
            }
            .padding(.top, 20)
            
            VStack(spacing: 12) {
                Text("Welcome to Aura")
                    .font(.system(size: 32, weight: .bold, design: .rounded))
                    .foregroundColor(.white)
                
                Text("A native, premium macOS environment for automated exploratory data analysis, machine learning modeling, and semantic insights.")
                    .font(.body)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 420)
                    .lineSpacing(4)
            }
            
            Spacer()
        }
        .padding(.horizontal, 40)
    }
    
    private var featuresSlide: some View {
        VStack(spacing: 24) {
            Text("All-in-One Data Toolkit")
                .font(.system(size: 24, weight: .bold, design: .rounded))
                .foregroundColor(.white)
                .padding(.top, 10)
            
            VStack(spacing: 16) {
                featureRow(
                    icon: "tablecells.fill",
                    color: .purple,
                    title: "Tabular & Regression / Classification",
                    desc: "Analyze correlations, evaluate model leaderboards, calculate SHAP feature importance, and export production-ready reproduction code."
                )
                
                featureRow(
                    icon: "text.justify.left",
                    color: .green,
                    title: "Natural Language Processing (NLP)",
                    desc: "Vectorize text features using TF-IDF and fit classifications or regression tasks directly on document collections."
                )
                
                featureRow(
                    icon: "photo.stack.fill",
                    color: .orange,
                    title: "Semantic Image Segmentation",
                    desc: "Train Random Forest pixel segmentations. Preview Dice/IoU metrics, mask overlays, and compare results instantly."
                )
            }
            
            Spacer()
        }
        .padding(.horizontal, 40)
    }
    
    private var timeSeriesSlide: some View {
        VStack(spacing: 24) {
            ZStack {
                Circle()
                    .fill(Color.blue.opacity(0.12))
                    .frame(width: 80, height: 80)
                
                Image(systemName: "chart.line.uptrend.xyaxis")
                    .font(.system(size: 36, weight: .bold))
                    .foregroundColor(.blue)
            }
            .padding(.top, 10)
            
            VStack(spacing: 12) {
                Text("Time Series & Forecasting")
                    .font(.system(size: 24, weight: .bold, design: .rounded))
                    .foregroundColor(.white)
                
                Text("Forecast multiple targets simultaneously with ARIMA and Linear models. Filter chronologically by year, explore calendar-seasonal cycles, and view clean continuous charts without label overlaps.")
                    .font(.body)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 440)
                    .lineSpacing(4)
            }
            
            Spacer()
        }
        .padding(.horizontal, 40)
    }
    
    private var aiSlide: some View {
        VStack(spacing: 24) {
            ZStack {
                Circle()
                    .fill(Color.purple.opacity(0.12))
                    .frame(width: 80, height: 80)
                
                Image(systemName: "sparkles")
                    .font(.system(size: 36, weight: .bold))
                    .foregroundColor(.purple)
            }
            .padding(.top, 10)
            
            VStack(spacing: 12) {
                Text("Secure AI Analyst Panel")
                    .font(.system(size: 24, weight: .bold, design: .rounded))
                    .foregroundColor(.white)
                
                Text("Leverage local Ollama LLMs directly on your machine. Chat with your dataset, ask for statistical advice, and generate comprehensive Markdown review reports without sending your data to the cloud.")
                    .font(.body)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 440)
                    .lineSpacing(4)
            }
            
            Spacer()
        }
        .padding(.horizontal, 40)
    }
    
    // MARK: - Slide Helpers
    
    private func featureRow(icon: String, color: Color, title: String, desc: String) -> some View {
        HStack(alignment: .top, spacing: 14) {
            ZStack {
                RoundedRectangle(cornerRadius: 8)
                    .fill(color.opacity(0.12))
                    .frame(width: 36, height: 36)
                
                Image(systemName: icon)
                    .font(.system(size: 16))
                    .foregroundColor(color)
            }
            
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.system(.body, design: .rounded))
                    .fontWeight(.semibold)
                    .foregroundColor(.primary)
                
                Text(desc)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(3)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
        }
        .padding(10)
        .background(Color.primary.opacity(0.01))
        .cornerRadius(10)
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.primary.opacity(0.04), lineWidth: 1))
    }
    
    // MARK: - Bottom Panel
    
    private var bottomPanel: some View {
        VStack(spacing: 16) {
            Divider().background(Color.primary.opacity(0.06))
            
            HStack {
                // Page Indicator Dots
                HStack(spacing: 6) {
                    ForEach(0..<totalPages, id: \.self) { index in
                        Circle()
                            .fill(index == currentPage ? Color.purple : Color.secondary.opacity(0.3))
                            .frame(width: index == currentPage ? 8 : 6, height: index == currentPage ? 8 : 6)
                            .animation(.spring(), value: currentPage)
                    }
                }
                
                Spacer()
                
                // Back Button (hidden on first page)
                if currentPage > 0 {
                    Button("Back") {
                        withAnimation {
                            currentPage -= 1
                        }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.regular)
                    .transition(.opacity)
                }
                
                // Next / Get Started Button
                Button {
                    if currentPage < totalPages - 1 {
                        withAnimation {
                            currentPage += 1
                        }
                    } else {
                        isPresented = false
                    }
                } label: {
                    Text(currentPage == totalPages - 1 ? "Get Started" : "Continue")
                        .fontWeight(.semibold)
                        .foregroundColor(.white)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 8)
                        .background(Theme.Color.brandGradient)
                        .cornerRadius(Theme.Layout.cornerRadius)
                }
                .buttonStyle(.plain)
                .shadow(color: .purple.opacity(0.25), radius: 6, x: 0, y: 3)
            }
            .padding(.horizontal, 32)
            .padding(.bottom, 24)
        }
    }
}

#Preview {
    OnboardingView(isPresented: .constant(true))
}
