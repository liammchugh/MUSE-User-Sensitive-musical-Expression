import SwiftUI

@main
struct SensorWatchApp: App {
    // One shared instance for the whole UI
    @StateObject private var dataManager = SensorDataManager()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(dataManager)
        }
    }
}
