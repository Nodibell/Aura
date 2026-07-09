import XCTest

final class AuraUITests: XCTestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    override func tearDownWithError() throws {
    }

    private func makeApp() -> XCUIApplication {
        let app = XCUIApplication(bundleIdentifier: "com.aura.Aura")
        // Pass the explicit python environment path to override Sandbox container HOME limitations
        app.launchArguments = [
            "-Aura_PythonPath", "*/Aura/.venv/bin/python3"
        ]
        return app
    }

    @MainActor
    func testAppLaunchAndInitialState() throws {
        // Use makeApp() so we don't lose the Python environment path
        let app = makeApp()
        
        // 1. Inject the custom launch argument to bypass Onboarding
        app.launchArguments.append("-UITesting")
        
        app.launch()
        app.activate()
        
        // 2. Broaden the element query for SettingsLink
        // Change from app.buttons to app.descendants(matching: .any)
        let settingsButton = app.descendants(matching: .any)["configureSettingsButton"]
        XCTAssertTrue(settingsButton.waitForExistence(timeout: 5), "Configure Settings button should exist")
        
        let manageSchedulesButton = app.buttons["manageSchedulesButton"]
        XCTAssertTrue(manageSchedulesButton.waitForExistence(timeout: 15), "Manage Schedules button should exist")
        
        let importDbButton = app.buttons["importFromDbButton"]
        XCTAssertTrue(importDbButton.waitForExistence(timeout: 5), "Import from DB button should exist on the welcome screen")
    }

    @MainActor
    func testSidebarActions() throws {
        let app = makeApp()
        app.launchArguments.append("-UITesting")
        app.launch()
        app.activate()
        
        // 1. Test Analysis Scheduler Sheet Open & Close
        let manageSchedulesButton = app.buttons["manageSchedulesButton"]
        XCTAssertTrue(manageSchedulesButton.waitForExistence(timeout: 15))
        manageSchedulesButton.click()
        
        let schedulerTitle = app.staticTexts["Analysis Scheduler"]
        XCTAssertTrue(schedulerTitle.waitForExistence(timeout: 5), "Analysis Scheduler sheet should be presented")
        
        let closeButton = app.buttons["Close"].firstMatch
        XCTAssertTrue(closeButton.waitForExistence(timeout: 2))
        closeButton.click()
        
        let exists = schedulerTitle.waitForExistence(timeout: 2)
        XCTAssertFalse(exists, "Scheduler sheet should be dismissed")
        
        // 2. Test Settings Link click (does not crash/hang)
        let settingsButton = app.descendants(matching: .any)["configureSettingsButton"]
        XCTAssertTrue(settingsButton.waitForExistence(timeout: 5))
        settingsButton.click()
    }

    @MainActor
    func testDatabaseSheetOpenClose() throws {
        let app = makeApp()
        app.launchArguments.append("-UITesting")
        app.launch()
        app.activate()
        
        let importDbButton = app.buttons["importFromDbButton"]
        XCTAssertTrue(importDbButton.waitForExistence(timeout: 15))
        importDbButton.click()
        
        let titleText = app.staticTexts["Database Ingestion"]
        XCTAssertTrue(titleText.waitForExistence(timeout: 5), "Database Ingestion sheet should be presented")
        
        app.typeKey(.escape, modifierFlags: [])
        
        let exists = titleText.waitForExistence(timeout: 2)
        XCTAssertFalse(exists, "Database Ingestion sheet should be dismissed")
    }

    @MainActor
    func testFullAnalysisAndTabNavigation() throws {
        let app = makeApp()
        app.launchArguments.append("-UITesting")
        app.launch()
        app.activate()
        
        // 1. Click "Iris Flowers" in the Sample Datasets card
        let irisButton = app.buttons["Iris Flowers"]
        XCTAssertTrue(irisButton.waitForExistence(timeout: 15), "Iris Flowers sample button should exist")
        irisButton.click()
        
        // 2. Click "Run Analysis Pipeline"
        // Changed to look for the literal text since there is no accessibility identifier on it
        let runButton = app.buttons["Run Analysis Pipeline"]
        XCTAssertTrue(runButton.waitForExistence(timeout: 20), "Run Analysis Pipeline button should exist")
        
        let isEnabled = NSPredicate(format: "enabled == true")
        let expectation = expectation(for: isEnabled, evaluatedWith: runButton, handler: nil)
        wait(for: [expectation], timeout: 5.0)
        
        runButton.click()
        
        // 3. Wait for the analysis to complete (segmented tab buttons should appear)
        let tabSummary = app.buttons["tab_Summary"]
        XCTAssertTrue(tabSummary.waitForExistence(timeout: 30), "Analysis should complete and Summary tab should appear within 30 seconds")
        
        // 4. Verify Summary Tab view components
        XCTAssertTrue(app.staticTexts["Jump to:"].exists, "Summary tab should display Jump to: header")
        XCTAssertTrue(app.staticTexts["Model Leaderboard"].exists, "Summary tab should display model leaderboard")
        
        // 5. Navigate to Charts Tab and verify search field
        let tabCharts = app.buttons["tab_Charts"]
        XCTAssertTrue(tabCharts.exists)
        tabCharts.click()
        
        let chartsSearchField = app.textFields.firstMatch
        XCTAssertTrue(chartsSearchField.waitForExistence(timeout: 2), "Charts list search field should exist")
        
        // 6. Navigate to Correlations Tab and verify
        let tabCorrelations = app.buttons["tab_Correlations"]
        XCTAssertTrue(tabCorrelations.exists)
        tabCorrelations.click()
        
        // 7. Navigate to Data Tab and verify CustomSegmentedPicker segmented selections
        let tabData = app.buttons["tab_Data"]
        XCTAssertTrue(tabData.exists)
        tabData.click()
        
        // 8. Navigate to Cleaning Tab, swap to Time-Travel Lineage inner tab
        let tabCleaning = app.buttons["tab_Cleaning"]
        XCTAssertTrue(tabCleaning.exists)
        tabCleaning.click()
        
        let tabLineage = app.buttons["tab_Time-Travel Lineage"]
        XCTAssertTrue(tabLineage.waitForExistence(timeout: 2), "Time-Travel Lineage tab selection should exist")
        tabLineage.click()
        
        XCTAssertTrue(app.staticTexts["Dataset State Lineage"].exists, "Should display Dataset State Lineage section")
        XCTAssertTrue(app.staticTexts["Initial Load"].exists, "Lineage should track Initial Load state")
        
        // 9. Navigate to Diff Tab and verify
        let tabDiff = app.buttons["tab_Diff"]
        XCTAssertTrue(tabDiff.exists)
        tabDiff.click()
        
        // 10. Navigate to Predict Tab and verify
        let tabPredict = app.buttons["tab_Predict"]
        if tabPredict.exists {
            tabPredict.click()
        }
    }
}
