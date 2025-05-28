import socket
import struct
import time

def build_tcp_packet(data_points, device_id=111, cmd_type=1, start_addr=0, packet_num=1):
    """Build TCP packet in same format as server expects"""
    m_send_dataNum = len(data_points)
    total_bytes = 7 + 4 * m_send_dataNum

    Head = bytearray(13)
    Head[0] = 1
    Head[1] = packet_num
    Head[2] = 0
    Head[3] = 0
    Head[4] = total_bytes & 0xFF
    Head[5] = (total_bytes >> 8) & 0xFF
    Head[6] = device_id
    Head[7] = cmd_type
    Head[8] = 0
    Head[9] = (start_addr) & 0xFF
    Head[10] = m_send_dataNum & 0xFF
    Head[11] = (m_send_dataNum >> 8) & 0xFF
    Head[12] = 0

    data_section = b''.join(struct.pack('<f', float(val)) for val in data_points)
    return Head + data_section

def main():
    HOST = '127.0.0.1'
    PORT = 12345
    RECONNECT_DELAY = 5  # seconds
    
    # Example ship position data (longitude, latitude, heading)
    ship_data = [46.77, 120.0, 45.6678, 30.0, 33.2243]  # Last value is heading angle
    
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5.0)  # Set socket timeout
                print(f"Attempting to connect to {HOST}:{PORT}...")
                
                try:
                    s.connect((HOST, PORT))
                    print(f"Connected to {HOST}:{PORT}")
                    
                    while True:
                        try:
                            # Send ship position data
                            packet = build_tcp_packet(ship_data)
                            s.sendall(packet)
                            print(f"Sent position data: {ship_data}")
                            
                            # Receive shoreline data (if any)
                            try:
                                data = s.recv(1024)
                                if data:
                                    try:
                                        # Parse header (13 bytes)
                                        header = data[:13]
                                        # Parse data points (remaining bytes as float32)
                                        data_points = data[13:]
                                        num_floats = len(data_points) // 4
                                        if num_floats > 0:
                                            values = struct.unpack(f'<{num_floats}f', data_points)
                                            print(f"Received shoreline data: {values}")
                                        else:
                                            print("Received empty data packet")
                                    except Exception as e:
                                        print(f"Error parsing data: {e}")
                                else:
                                    print("Server closed connection")
                                    break
                            except socket.timeout:
                                print("No data received, continuing...")
                                continue
                            
                            time.sleep(1)  # Send updates every second
                            
                        except ConnectionResetError:
                            print("Connection reset by peer")
                            break
                        except socket.timeout:
                            print("Connection timeout")
                            break
                        except Exception as e:
                            print(f"Communication error: {e}")
                            break
                            
                except ConnectionRefusedError:
                    print(f"Connection refused, retrying in {RECONNECT_DELAY} seconds...")
                    time.sleep(RECONNECT_DELAY)
                    continue
                except Exception as e:
                    print(f"Connection error: {e}")
                    time.sleep(RECONNECT_DELAY)
                    continue
                    
        except KeyboardInterrupt:
            print("\nClient shutting down...")
            break

if __name__ == '__main__':
    main()
