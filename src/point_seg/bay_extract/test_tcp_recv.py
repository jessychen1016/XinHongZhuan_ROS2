import socket
import struct

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
    if device_id not in (0x6f, 0xC8):  # 100 or 200
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

def main():
    HOST = '10.17.81.16'
    PORT = 12345

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"Listening on {HOST}:{PORT}")
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            while True:
                header = recv_all(conn, 13)
                iLength = header[4] | (header[5] << 8)
                total_len = iLength + 6
                if total_len > 13:
                    rest = recv_all(conn, total_len - 13)
                    packet = header + rest
                else:
                    packet = header
                parse_packet(packet)

if __name__ == "__main__":
    main()