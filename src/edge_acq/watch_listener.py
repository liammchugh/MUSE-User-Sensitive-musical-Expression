import socket, csv, datetime

UDP_IP   = "0.0.0.0"   # listen on all interfaces
# make sure Exporter IP Address matches acquisition IP Address
UDP_PORT = 9000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening on UDP {UDP_PORT}…  Ctrl‑C to stop")
with open("watch_stream.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["unix_ts","ax","ay","az","gx","gy","gz","bpm","tempC"])

    while True:
        data, addr = sock.recvfrom(1024)
        line = data.decode().strip()
        writer.writerow(line.split(","))
