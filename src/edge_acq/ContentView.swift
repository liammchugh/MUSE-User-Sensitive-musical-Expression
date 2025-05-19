struct ContentView: View {
    @EnvironmentObject var data: SensorDataManager
    @State private var showSettings = false
    @State private var draftHost = ""

    var body: some View {
        VStack(spacing: 6) {
            HStack {
                Spacer()
                Button(action: { draftHost = data.udpHost; showSettings = true }) {
                    Image(systemName: "gearshape.fill")
                }
                .sheet(isPresented: $showSettings) { settingsSheet }
            }
            .padding(.trailing, 4)

            ScrollView { sensorList }
            exportButton
            if data.isExporting { statusBanner }
        }
        .padding()
    }

    // MARK: -- Settings Sheet
    private var settingsSheet: some View {
        VStack(spacing: 12) {
            Text("UDP Destination").font(.headline)

            TextField("Host or IP", text: $draftHost)
                .textContentType(.URL)
                .multilineTextAlignment(.center)
                .submitLabel(.done)
                .onSubmit(saveHost)

            Button("Save", action: saveHost)
                .buttonStyle(.borderedProminent)

            Button("Cancel") { showSettings = false }
                .foregroundColor(.red)
        }
        .padding()
    }

    private func saveHost() {
        data.udpHost = draftHost.trimmingCharacters(in: .whitespaces)
        showSettings = false
    }

    @ViewBuilder
    private var sensorList: some View {
        Group {
            // Accelerometer section
            if let a = data.accel?.acceleration {
                Text("Accel  x:\(format(a.x)) y:\(format(a.y)) z:\(format(a.z))")
            }
            // Gyro section
            if let g = data.gyro?.rotationRate {
                Text("Gyro   x:\(format(g.x)) y:\(format(g.y)) z:\(format(g.z))")
            }
            // Heart + Temp
            Text("HR \(format(data.heartRateBPM)) BPM")
            if let t = data.wristTempC {
                Text("Temp \(format(t)) °C")
            }
        }
        .font(.system(size: 14, design: .monospaced))
    }

    private func format(_ v: Double) -> String {
        v.isNaN ? "—" : String(format: "%.2f", v)
    }
}
