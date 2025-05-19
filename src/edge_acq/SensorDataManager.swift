import Foundation
import CoreMotion
import HealthKit
import Combine

final class SensorDataManager: NSObject, ObservableObject {

    // MARK: ‑‑ Publicly observed values
    @Published var accel: CMAccelerometerData?
    @Published var gyro: CMGyroData?
    @Published var heartRateBPM: Double = 0          // live stream
    @Published var wristTempC: Double?               // one‑shot, optional

    // MARK: ‑‑ Exporting
    @Published var isExporting = false
    private let exporter = SensorDataExporter()
    private var runtimeSession: WKExtendedRuntimeSession?

    // MARK: ‑‑ Private stores
    private let motionMgr = CMMotionManager()
    private let healthStore = HKHealthStore()
    private var workoutSession: HKWorkoutSession?
    private var workoutBuilder: HKLiveWorkoutBuilder?
    private var cancellables = Set<AnyCancellable>()

    // MARK: UserDefaults key
    private static let hostKey = "udpHost"
    // Published so UI reflects changes instantly
    @Published var udpHost: String {
        didSet {
            exporter.updateHost(udpHost)
            UserDefaults.standard.set(udpHost, forKey: Self.hostKey)
        }
    }

    // MARK: ‑‑ Init
    override init() {
        let saved = UserDefaults.standard.string(forKey: Self.hostKey) ?? "192.168.1.4"
        self.udpHost = saved
        self.exporter = SensorDataExporter(savedHost: saved)
        super.init()
        authoriseHealthKit()
        startMotion()
    }

    // MARK: ‑‑ Motion
    private func startMotion() {
        guard motionMgr.isAccelerometerAvailable,
              motionMgr.isGyroAvailable else {
            print("Motion sensors unavailable")
            return
        }

        motionMgr.accelerometerUpdateInterval = 1.0/50.0  // 50 Hz
        motionMgr.gyroUpdateInterval          = 1.0/50.0

        motionMgr.startAccelerometerUpdates(to: .main) { [weak self] acc, _ in
            guard let self, let acc else { return }
            self.accel = acc
            self.emitIfExporting()
        }
        motionMgr.startGyroUpdates(to: .main) { [weak self] gyr, _ in
            guard let self, let gyr else { return }
            self.gyro = gyr
        }
    }

    // ------------- Export helpers -------------
    func toggleExport() {
        if isExporting { stopExport() } else { startExport() }
    }

    private func startExport() {
        exporter.start()
        isExporting = true

        // Keep the app alive after the screen turns off
        runtimeSession = WKExtendedRuntimeSession()
        runtimeSession?.start()
    }

    private func stopExport() {
        exporter.stop()
        isExporting = false
        runtimeSession?.invalidate()
        runtimeSession = nil
    }
    
    private func emitIfExporting() {
        guard isExporting,
              let acc = accel?.acceleration,
              let gyr = gyro?.rotationRate else { return }

        let ts = Date().timeIntervalSince1970
        let bpm = heartRateBPM
        let temp = wristTempC ?? .nan
        let line = String(format: "%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f,%.2f\n",
                          ts,
                          acc.x, acc.y, acc.z,
                          gyr.x, gyr.y, gyr.z,
                          bpm,
                          temp)
        exporter.send(line: line)
    }

    // MARK: ‑‑ HealthKit
    private func authoriseHealthKit() {
        guard HKHealthStore.isHealthDataAvailable() else { return }

        let heart = HKQuantityType.quantityType(forIdentifier: .heartRate)!
        let temp  = HKQuantityType.quantityType(forIdentifier: .bodyTemperature)!

        healthStore.requestAuthorization(toShare: [heart],  // allow workout samples
                                         read:   [heart, temp]) { [weak self] ok, err in
            guard ok else { print("HK auth failed:", err ?? ""); return }
            DispatchQueue.main.async {
                self?.startWorkoutSession()
                self?.fetchLatestTemperature()
            }
        }
    }

    // Continuous heart‑rate: keep a workout running in the foreground.
    private func startWorkoutSession() {
        let config = HKWorkoutConfiguration()
        config.activityType = .other
        config.locationType = .unknown

        do {
            workoutSession = try HKWorkoutSession(healthStore: healthStore,
                                                  configuration: config)
            workoutBuilder = workoutSession!.associatedWorkoutBuilder()
            workoutBuilder!.dataSource = HKLiveWorkoutDataSource(healthStore: healthStore,
                                                                 workoutConfiguration: config)

            // Listen for new HR samples
            workoutBuilder!.statisticsCollectionPublisher
                .receive(on: RunLoop.main)
                .sink { [weak self] stats in
                    guard let hrStats = stats.statistics(for: .heartRate),
                          let bpmUnit = HKUnit.count().unitDivided(by: .minute()),
                          let value = hrStats.mostRecentQuantity()?.doubleValue(for: bpmUnit)
                    else { return }
                    self?.heartRateBPM = value
                }
                .store(in: &cancellables)

            // Start the session
            workoutSession!.startActivity(with: .now)
            workoutBuilder!.beginCollection(withStart: .now) { _, _ in }
        } catch {
            print("Workout start error:", error)
        }
    }

    // One‑shot: get the newest wrist‑temperature (if any)
    private func fetchLatestTemperature() {
        guard let tempType = HKQuantityType.quantityType(forIdentifier: .bodyTemperature) else { return }

        let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        let q = HKSampleQuery(sampleType: tempType,
                              predicate: nil,
                              limit: 1,
                              sortDescriptors: [sort]) { [weak self] _, samples, _ in
            guard let qSample = samples?.first as? HKQuantitySample else { return }
            let val = qSample.quantity.doubleValue(for: .degreeCelsius())
            DispatchQueue.main.async { self?.wristTempC = val }
        }
        healthStore.execute(q)
    }
}
