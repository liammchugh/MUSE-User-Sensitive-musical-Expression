import Foundation
import CoreMotion
import HealthKit
import Combine
import WatchKit

final class SensorDataManager: NSObject, ObservableObject {

    // MARK: – Publicly observed values
    @Published var accel: CMAccelerometerData?
    @Published var gyro:  CMGyroData?
    @Published var heartRateBPM: Double = 0
    @Published var wristTempC:  Double?

    // MARK: – Exporting
    @Published var isExporting = false
    private var exporter: SensorDataExporter      // single instance
    private var runtimeSession: WKExtendedRuntimeSession?

    // MARK: – Private stores
    private let motionMgr   = CMMotionManager()
    private let healthStore = HKHealthStore()
    private var workoutSession:  HKWorkoutSession?
    private var workoutBuilder:  HKLiveWorkoutBuilder?
    private var cancellables     = Set<AnyCancellable>()

    // MARK: – Host persistence
    private static let hostKey = "udpHost"
    @Published var udpHost: String {
        didSet {
            exporter.updateHost(udpHost)
            UserDefaults.standard.set(udpHost, forKey: Self.hostKey)
        }
    }

    // MARK: – Init
    override init() {
        let saved = UserDefaults.standard.string(forKey: Self.hostKey) ?? "192.168.1.4"
        udpHost   = saved
        exporter  = SensorDataExporter(savedHost: saved)
        super.init()
        authoriseHealthKit()
        startMotion()
    }

    // MARK: – Motion
    private func startMotion() {
        guard motionMgr.isAccelerometerAvailable,
              motionMgr.isGyroAvailable else {
            print("Motion sensors unavailable")
            return
        }

        motionMgr.accelerometerUpdateInterval = 1/50.0
        motionMgr.gyroUpdateInterval          = 1/50.0

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

    // MARK: – Export helpers
    func toggleExport() { isExporting ? stopExport() : startExport() }

    private func startExport() {
        exporter.start(); isExporting = true
        runtimeSession = WKExtendedRuntimeSession(); runtimeSession?.start()
    }

    private func stopExport() {
        exporter.stop();  isExporting = false
        runtimeSession?.invalidate(); runtimeSession = nil
    }

    private func emitIfExporting() {
        guard isExporting,
              let a = accel?.acceleration,
              let g = gyro?.rotationRate else { return }

        let line = String(format: "%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f,%.2f\n",
                          Date().timeIntervalSince1970,
                          a.x, a.y, a.z,
                          g.x, g.y, g.z,
                          heartRateBPM,
                          wristTempC ?? .nan)
        exporter.send(line: line)
    }

    // MARK: – HealthKit Authorisation
    private func authoriseHealthKit() {
        guard HKHealthStore.isHealthDataAvailable() else { return }

        let heart = HKQuantityType.quantityType(forIdentifier: .heartRate)!
        let temp  = HKQuantityType.quantityType(forIdentifier: .bodyTemperature)!

        healthStore.requestAuthorization(toShare: [heart], read: [heart, temp]) { [weak self] ok, err in
            guard ok else { print("HK auth failed:", err ?? ""); return }
            DispatchQueue.main.async {
                self?.startWorkoutSession()
                self?.fetchLatestTemperature()
            }
        }
    }

    // MARK: – Workout session / live heart-rate
    // MARK: – Workout session / live heart-rate  (delegate only → works on watchOS 5-11)
    private func startWorkoutSession() {

        // This code is excluded when you compile for the watchOS Simulator,
        // which avoids the “Watch-Only stubs” and ‘heartRate’ symbol errors.
        #if !targetEnvironment(simulator)

        let config = HKWorkoutConfiguration()
        config.activityType = .other
        config.locationType = .unknown

        do {
            workoutSession = try HKWorkoutSession(healthStore: healthStore,
                                                  configuration: config)
            workoutBuilder = workoutSession!.associatedWorkoutBuilder()
            workoutBuilder!.dataSource = HKLiveWorkoutDataSource(healthStore: healthStore,
                                                                 workoutConfiguration: config)

            // Just the delegate — no Combine, no statisticsCollectionPublisher
            workoutBuilder!.delegate = self

            workoutSession!.startActivity(with: .now)
            workoutBuilder!.beginCollection(withStart: .now) { _, _ in }
        } catch {
            print("Workout start error:", error)
        }

        #endif
    }


    // MARK: – One-shot wrist temperature
    private func fetchLatestTemperature() {
        guard let tempType = HKQuantityType.quantityType(forIdentifier: .bodyTemperature) else { return }
        let q = HKSampleQuery(sampleType: tempType,
                              predicate: nil,
                              limit: 1,
                              sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)]) {
            [weak self] _, samples, _ in
            if let qSample = samples?.first as? HKQuantitySample {
                let val = qSample.quantity.doubleValue(for: .degreeCelsius())
                DispatchQueue.main.async { self?.wristTempC = val }
            }
        }
        healthStore.execute(q)
    }
}

#if !targetEnvironment(simulator)
extension SensorDataManager: HKLiveWorkoutBuilderDelegate {

    func workoutBuilder(_ builder: HKLiveWorkoutBuilder,
                        didCollectDataOf collectedTypes: Set<HKSampleType>) {

        guard collectedTypes.contains(where: {
            ($0 as? HKQuantityType) ==
               HKQuantityType.quantityType(forIdentifier: .heartRate)
        }) else { return }

        if let bpm = builder.statistics(for: .heartRate)?
                        .mostRecentQuantity()?
                        .doubleValue(for:
                           .count().unitDivided(by: .minute())) {

            DispatchQueue.main.async { self.heartRateBPM = bpm }
        }
    }

    func workoutBuilderDidCollectEvent(_ builder: HKLiveWorkoutBuilder) { }
}
#endif

