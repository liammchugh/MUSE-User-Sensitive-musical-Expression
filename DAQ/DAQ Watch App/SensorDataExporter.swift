import Foundation
import Network

/// Very small helper that streams CSV lines over UDP, 64-line FIFO Buffer
final class SensorDataExporter {
    
    // MARK: ‑‑ Public state
    
    // MARK: ‑‑ Private
    private var connection: NWConnection?
    private(set) var host: NWEndpoint.Host
    private let port: NWEndpoint.Port = 9000
    private(set) var isRunning = false
    
    init(savedHost: String) {
        guard !savedHost.isEmpty else { fatalError("Host must not be empty") }
        host = NWEndpoint.Host(savedHost)
    }
    
    
    // NEW - update host & reconnect if needed
    func updateHost(_ newHost: String) {
        host = NWEndpoint.Host(newHost)
        if isRunning {
            stop()
            start()
        }
    }
    
    func start() {
        guard !isRunning else { return }
        
        connection = NWConnection(host: host, port: port, using: .udp)
        connection?.stateUpdateHandler = { [weak self] state in
            guard let self, let conn = self.connection else { return }
            
            switch state {
            case .ready:
                // UDP socket is ready → push any queued packets
                self.flushBuffer(to: conn)
                
            case .failed(let err):
                print("UDP failed:", err)
                self.stop()              // mark not running; ring keeps accumulating
                
            case .waiting(let err):
                print("UDP waiting:", err)   // e.g. network down; keep buffering
                
            default:
                break
            }
        }
        connection?.start(queue: .main)
        isRunning = true
    }
    
    
    func stop() {
        guard isRunning else { return }
        connection?.cancel()
        connection = nil
        isRunning = false
    }
    
    /// Send  CSV record, e.g. `"timestamp,ax,ay,az,gx,gy,gz,bpm,temp\n"`
    /// uses 64-line FIFO buffer queue
    private var ring = [Data](repeating: Data(), count: 64)
    private var writeIdx = 0           // where the next write will go
    private var readIdx  = 0           // oldest packet not yet sent
    
    func send(line: String) {
        guard let packet = line.data(using: .utf8) else { return }
        
        if isRunning, let conn = connection {
            conn.send(content: packet, completion: .idempotent)
        } else {
            ring[writeIdx % ring.count] = packet
            writeIdx += 1
            
            // Overwrite oldest if buffer is full
            if writeIdx - readIdx > ring.count {
                readIdx += 1
            }
        }
    }
    
    private func flushBuffer(to conn: NWConnection) {
        while readIdx < writeIdx {
            let packet = ring[readIdx % ring.count]
            conn.send(content: packet, completion: .idempotent)
            readIdx += 1
        }
        // Reset indices so they don't grow without bound
        readIdx = 0
        writeIdx = 0
    }
    
}
