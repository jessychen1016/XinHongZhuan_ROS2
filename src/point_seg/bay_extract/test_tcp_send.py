import socket
import struct
import time

def recv_all(sock, length):
    data = b''
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            raise EOFError('Socket closed before we received all data')
        data += more
    return data

def parse_packet(packet):
    if len(packet) < 13:
        print("Packet too short")
        return

    buf = packet
    device_id = buf[6]
    if device_id not in (0x6f, 0xC8):  # 111 or 200
        print(f"Invalid device ID: {device_id}")
        return

    iLength = buf[4] | (buf[5] << 8)
    if iLength + 6 > len(packet):
        print("Packet length mismatch")
        return

    iDataNum = buf[10] | (buf[11] << 8)
    if (iDataNum * 4) + 7 != iLength:
        print("Data length does not match header")
        return

    data_start = 13
    data_end = data_start + iDataNum * 4
    data_section = packet[data_start:data_end]
    data_points = [
        struct.unpack('<f', data_section[i*4:(i+1)*4])[0]
        for i in range(iDataNum)
    ]
    print(f"Received data points: {data_points}")

def build_packet(data_points, device_id=111, cmd_type=1, start_addr=0, packet_num=1):
    m_send_dataNum = len(data_points)
    total_bytes = 7 + 4 * m_send_dataNum  # header + data as per protocol

    # Build header (13 bytes)
    Head = bytearray(13)
    Head[0] = 1
    Head[1] = packet_num  # Communication number
    Head[2] = 0
    Head[3] = 0  # TCP protocol type
    Head[4] = total_bytes & 0xFF
    Head[5] = (total_bytes >> 8) & 0xFF
    Head[6] = device_id  # Device address
    Head[7] = cmd_type   # Command type
    Head[8] = 0
    Head[9] = (start_addr) & 0xFF
    Head[10] = m_send_dataNum & 0xFF
    Head[11] = (m_send_dataNum >> 8) & 0xFF
    Head[12] = 0  # Reserved

    # Pack all data points as float32 (little-endian)
    data_section = b''.join(struct.pack('<f', float(val)) for val in data_points)
    return Head + data_section

def main():
    # Example: 27 float values (replace with your real data)
    base_data = [
        60.0, 90.78, 90.12,   # 1-3: distance
        45.67,                 # 4: drift angle
        111.1, 222.2, 333.3,   # 5-7: bow/mid/stern speed
        45,                     # 8: bow heading angle
        10.5, 3.14,            # 9-10: rotation speed, yaw angle
        2,                     # 11: shoreline point count
        30, 15, 1,             #  1# lat-deg/min/hemisphere
        120, 35, 1,            #  1# lon-deg/min/hemisphere
        1000, 2000,            #  1# distance to ship center
        25, 10, 2,             #  2# lat-deg/min/hemisphere
        130, 40, 2,            #  2# lon-deg/min/hemisphere
        1100, 2100             #  2# distance to ship center
        30, 15, 1,             #  3# lat-deg/min/hemisphere
        120, 35, 1,            #  3# lon-deg/min/hemisphere
        1000, 2000,            #  3# distance to ship center
        25, 10, 2,             #  4# lat-deg/min/hemisphere
        130, 40, 2,            #  4# lon-deg/min/hemisphere
        1100, 2100             #  4# distance to ship center
        30, 15, 1,             #  5# lat-deg/min/hemisphere
        120, 35, 1,            #  5# lon-deg/min/hemisphere
        1000, 2000,            #  5# distance to ship center
        25, 10, 2,             #  6# lat-deg/min/hemisphere
        130, 40, 2,            #  6# lon-deg/min/hemisphere
        1100, 2100             #  6# distance to ship center
        25, 10, 2,             #  7# lat-deg/min/hemisphere
        130, 40, 2,            #  7# lon-deg/min/hemisphere
        1100, 2100             #  7# distance to ship center
        25, 10, 2,             #  8# lat-deg/min/hemisphere
        130, 40, 2,            #  8# lon-deg/min/hemisphere
        1100, 2100             #  8# distance to ship center
        25, 10, 2,             #  fore point# lat-deg/min/hemisphere
        130, 40, 2,            #  fore point# lon-deg/min/hemisphere
        1100, 2100             #  fore point# distance to ship center
        25, 10, 2,             #  fore intercept# lat-deg/min/hemisphere
        130, 40, 2,            #  fore intercept# lon-deg/min/hemisphere
        1100, 2100             #  fore intercept# distance to ship center
        25, 10, 2,             #  after point# lat-deg/min/hemisphere
        130, 40, 2,            #  after point# lon-deg/min/hemisphere
        1100, 2100             #  after point# distance to ship center
        25, 10, 2,             #  after intercept# lat-deg/min/hemisphere
        130, 40, 2,            #  after intercept# lon-deg/min/hemisphere
        1100, 2100             #  after intercept# distance to ship center
    ]
    HOST = '127.0.0.1'
    PORT = 12345

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        while True:
            try:
                s.bind((HOST, PORT))
                break
            except OSError as e:
                if e.errno == 98:  # Address already in use
                    print(f"Port {PORT} in use, retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    raise
        s.listen(1)
        print(f"Waiting for connection on {HOST}:{PORT}")
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            packet_num = 1
            try:
                while True:
                    try:
                        data_points = [v + 0.01 * packet_num for v in base_data]
                        packet = build_packet(data_points, device_id=111, packet_num=packet_num)
                        conn.sendall(packet)
                        print(f"Sent packet {packet_num} of {len(packet)} bytes")
                        packet_num = (packet_num + 1) % 256
                        header = recv_all(conn, 13)
                        iLength = header[4] | (header[5] << 8)
                        total_len = iLength + 6
                        if total_len > 13:
                            rest = recv_all(conn, total_len - 13)
                            packet = header + rest
                        else:
                            packet = header
                        parse_packet(packet)
                        time.sleep(0.01)
                    except EOFError:
                        print("Client disconnected.")
                        break
                    except Exception as e:
                        print(f"Error in loop: {e}")
                        break
            except KeyboardInterrupt:
                print("Stopped sending.")

if __name__ == '__main__':
    main()