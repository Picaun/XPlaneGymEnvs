"""
XPlaneGym - XPlane Flight Simulator Connection Class

This class provides functionality for communicating with the XPlane flight simulator,
including data subscription, reading, and sending control commands.
"""

import struct
import socket
import threading
import datetime
import time
from typing import Tuple, List, Dict, Union, Optional


class XPlaneConnect:
    """XPlane Flight Simulator Connection Class
    
    Provides functionality to communicate with the XPlane flight simulator via UDP,
    for sending control commands and receiving flight data.
    """
    
    def __init__(self, ip: str = '127.0.0.1', port: int = 49000, timeout: float = 1.0) -> None:
        """Initialize XPlane connection
        
        Args:
            ip: IP address where XPlane is running, default is local 127.0.0.1
            port: XPlane UDP communication port, default is 49000
            timeout: Communication timeout in seconds, default is 1.0 second
        """
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self._subscribed_drefs = []
        self.current_dref_values = {}
        self.reverse_index = {}
        self._observe_thread = None
        self._stop_flag = False
    
    def subscribe_drefs(self, drefs: List[Tuple[str, int]]) -> None:
        """Subscribe to data references (DataRefs)
        
        Args:
            drefs: List of (DataRef, frequency) tuples
                Example: [("sim/flightmodel/position/latitude", 10)]
        """
        self._subscribed_drefs = drefs
        self.reverse_index = {i: dref[0] for i, dref in enumerate(self._subscribed_drefs)}
        self.current_dref_values = {dref[0]: {'value': None, 'timestamp': None} for dref in self._subscribed_drefs}
        
        self._create_observation_requests()
        self._start_observe_thread()
    
    def _create_observation_requests(self) -> None:
        """Create and send data subscription requests"""
        for i, dref_tuple in enumerate(self._subscribed_drefs):
            dref, freq = dref_tuple
            cmd = b'RREF'  # Request DREF
            msg = struct.pack("<4sxii400s", cmd, freq, i, dref.encode('utf-8'))
            try:
                self.sock.sendto(msg, (self.ip, self.port))
            except Exception as e:
                print(f"Error sending subscription request: {e}")
    
    def _start_observe_thread(self) -> None:
        """Start asynchronous data reception thread"""
        self._stop_flag = False
        if self._observe_thread is not None and self._observe_thread.is_alive():
            return
            
        self._observe_thread = threading.Thread(target=self._observe)
        self._observe_thread.daemon = True
        self._observe_thread.start()
    
    def _observe(self) -> None:
        """Asynchronously receive and process data packets"""
        # Set non-blocking socket
        self.sock.settimeout(0.1)
        
        while not self._stop_flag:
            try:
                data, addr = self.sock.recvfrom(16348)
                if len(data) < 5:
                    continue
                    
                header = data[0:4]
                if header != b'RREF':
                    continue
                
                # Check packet length
                remainder = (len(data) - 5) % 8
                if remainder != 0:
                    # If length is not a multiple of 8, adjust to nearest valid length
                    valid_length = len(data) - remainder
                    data = data[:valid_length]
                    if valid_length <= 5:
                        continue  # No valid data
                
                # Parse each value in the packet
                no_packets = int((len(data) - 5) / 8)
                for p_idx in range(no_packets):
                    try:
                        p_data = data[(5 + p_idx * 8):(5 + (p_idx + 1) * 8)]
                        if len(p_data) != 8:
                            continue
                            
                        idx, value = struct.unpack("<if", p_data)
                        if idx in self.reverse_index:
                            dref = self.reverse_index[idx]
                            self.current_dref_values[dref] = {
                                'value': value,
                                'timestamp': datetime.datetime.now()
                            }
                    except Exception:
                        continue
            except socket.timeout:
                # Timeout is normal, continue loop
                continue
            except Exception as e:
                print(f"Error in data reception thread: {e}")
                time.sleep(0.1)
                continue
    
    def stop(self) -> None:
        """Stop data reception thread"""
        self._stop_flag = True
        if self._observe_thread and self._observe_thread.is_alive():
            self._observe_thread.join(timeout=1.0)
    
    def get_dref(self, dref: str) -> float:
        """Get current value of a data reference
        
        Args:
            dref: Name of the data reference to query
            
        Returns:
            Current value of the data reference (float)
        """
        # First check if already subscribed
        if dref in self.current_dref_values and self.current_dref_values[dref]['value'] is not None:
            return self.current_dref_values[dref]['value']
        
        # Otherwise send a one-time request
        temp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        temp_socket.settimeout(self.timeout)
        
        try:
            # Generate a random index to avoid conflicts
            idx = 1000 + hash(dref) % 1000  # Use hash to generate index
            
            # Send request
            msg = struct.pack("<4sxii400s", b'RREF', 1, idx, dref.encode('utf-8'))
            temp_socket.sendto(msg, (self.ip, self.port))
            
            # Try to receive response
            for attempt in range(3):
                try:
                    data, addr = temp_socket.recvfrom(1024)
                    if len(data) < 9 or data[0:4] != b'RREF':
                        continue
                        
                    try:
                        recv_idx, value = struct.unpack("<if", data[5:13])
                        if recv_idx == idx:
                            return value
                    except struct.error:
                        continue
                except socket.timeout:
                    continue
                    
            # All attempts failed
            return 0.0
        finally:
            temp_socket.close()
    
    def send_dref(self, dref: str, value: float) -> None:
        """Set the value of a data reference
        
        Args:
            dref: Name of the data reference to set
            value: Value to set (float)
        """
        try:
            msg = struct.pack('<4sxf500s', b'DREF', value, dref.encode('UTF-8'))
            self.sock.sendto(msg, (self.ip, self.port))
        except Exception as e:
            print(f"Error sending DREF: {e}")
    
    def send_command(self, command: str) -> None:
        """Send simulator command
        
        Args:
            command: Command to execute
        """
        try:
            msg = struct.pack('<4sx500s', b'CMND', command.encode('utf-8'))
            self.sock.sendto(msg, (self.ip, self.port))
        except Exception as e:
            print(f"Error sending command: {e}")
    
    def send_position(self, lat: float, lon: float, alt: float, phi: float, theta: float, psi: float, ac: int = 0) -> None:
        """Set aircraft position and attitude
        
        Args:
            lat: Latitude (degrees)
            lon: Longitude (degrees)
            alt: Altitude (meters)
            phi: Roll angle (degrees)
            theta: Pitch angle (degrees)
            psi: True heading (degrees)
            ac: Aircraft index, 0 for user-controlled aircraft
        """
        try:
            msg = struct.pack('<4sxidddfff', b'VEHS', ac, lat, lon, alt, psi, theta, phi)
            self.sock.sendto(msg, (self.ip, self.port))
            # Send twice to ensure altitude is set correctly
            self.sock.sendto(msg, (self.ip, self.port))
        except Exception as e:
            print(f"Error sending position: {e}")
    
    def get_position(self) -> Tuple:
        """Get aircraft current position and attitude
        
        Returns:
            Tuple containing the following values:
            - Latitude (degrees)
            - Longitude (degrees)
            - Altitude (meters)
            - Ground height (meters)
            - Roll angle (degrees)
            - Pitch angle (degrees)
            - True heading (degrees)
            - East velocity (meters/second)
            - Up velocity (meters/second)
            - South velocity (meters/second)
            - Roll rate (radians/second)
            - Pitch rate (radians/second)
            - Yaw rate (radians/second)
        """
        # Use DataRef method to get position information, avoiding RPOS packet format issues
        try:
            # Get basic position
            lat = self.get_dref("sim/flightmodel/position/latitude")
            lon = self.get_dref("sim/flightmodel/position/longitude")
            alt = self.get_dref("sim/flightmodel/position/elevation")
            agl = self.get_dref("sim/flightmodel/position/y_agl")
            
            # Get attitude
            phi = self.get_dref("sim/flightmodel/position/phi")  # Roll angle
            theta = self.get_dref("sim/flightmodel/position/theta")  # Pitch angle
            psi = self.get_dref("sim/flightmodel/position/psi")  # Yaw angle
            
            # Get velocity
            vx = self.get_dref("sim/flightmodel/position/local_vx")  # East velocity
            vy = self.get_dref("sim/flightmodel/position/local_vy")  # Up velocity
            vz = self.get_dref("sim/flightmodel/position/local_vz")  # South velocity
            
            # Get angular rates
            p = self.get_dref("sim/flightmodel/position/P")  # Roll rate
            q = self.get_dref("sim/flightmodel/position/Q")  # Pitch rate
            r = self.get_dref("sim/flightmodel/position/R")  # Yaw rate
            
            return (lat, lon, alt, agl, phi, theta, psi, vx, vy, vz, p, q, r)
            
        except Exception as e:
            # If DataRef method fails, try using the original RPOS method
            temp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            temp_socket.settimeout(self.timeout)
            
            try:
                # Request position data
                msg = struct.pack('4sx10s', b'RPOS', b'100')
                temp_socket.sendto(msg, (self.ip, self.port))
                
                # Try to receive response
                for attempt in range(3):
                    try:
                        data, addr = temp_socket.recvfrom(1024)
                        
                        if len(data) < 4 or data[0:4] != b'RPOS':
                            continue
                            
                        # Try different parsing methods based on actual returned data length
                        try:
                            if len(data) >= 108:  # Expected complete length 4 + 13 * 8
                                values = struct.unpack("<xdddffffffffff", data[4:])
                                # Rearrange to make return order more logical
                                # lon, lat, alt, agl, theta, psi, phi, vx, vy, vz, p, q, r
                                lon, lat, alt, agl, theta, psi, phi, vx, vy, vz, p, q, r = values
                                return lat, lon, alt, agl, phi, theta, psi, vx, vy, vz, p, q, r
                            elif len(data) >= 69:  # Actual returned length
                                # Try to adapt to actual data format returned by XPlane
                                # Assume first 8 values are available, fill rest with 0
                                fmt_str = "<x" + "d" * ((len(data) - 4) // 8)
                                partial_values = struct.unpack(fmt_str, data[4:4+((len(data)-4)//8)*8])
                                
                                # Extract available values, fill rest with 0
                                values_list = list(partial_values)
                                while len(values_list) < 13:
                                    values_list.append(0.0)
                                    
                                # Rearrange order
                                if len(values_list) >= 8:
                                    lon, lat, alt, agl, theta, psi, phi = values_list[0:7]
                                    remaining = values_list[7:] + [0.0] * max(0, 13 - len(values_list))
                                    vx, vy, vz, p, q, r = remaining[0:6]
                                    return lat, lon, alt, agl, phi, theta, psi, vx, vy, vz, p, q, r
                        except struct.error:
                            continue
                    except socket.timeout:
                        continue
                        
                # All attempts failed, return zero values
                return tuple([0.0] * 13)
            finally:
                temp_socket.close()
    
    def send_ctrl(self, 
                  lat_control: float = 0.0, 
                  lon_control: float = 0.0, 
                  rudder_control: float = 0.0, 
                  throttle: float = 0.5, 
                  gear: int = 1, 
                  flaps: float = 0.0, 
                  speedbrakes: float = 0.0, 
                  park_brake: float = 0.0) -> None:
        """Send flight control commands
        
        Args:
            lat_control: Lateral control (ailerons), range [-1...1]
            lon_control: Longitudinal control (elevator), range [-1...1]
            rudder_control: Rudder control, range [-1...1]
            throttle: Throttle position, range [0...1]
            gear: Landing gear position, 0=up, 1=down
            flaps: Flaps position, range [0...1]
            speedbrakes: Speed brakes position, range [0...1] or -0.5 for armed
            park_brake: Parking brake, range [0...1]
        """
        # Ensure all values are within valid range
        lat_control = max(-1.0, min(1.0, lat_control))
        lon_control = max(-1.0, min(1.0, lon_control))
        rudder_control = max(-1.0, min(1.0, rudder_control))
        throttle = max(0.0, min(1.0, throttle))
        gear = 1 if gear else 0
        flaps = max(0.0, min(1.0, flaps))
        
        try:
            # Lateral control (ailerons)
            self.send_dref("sim/cockpit2/controls/yoke_roll_ratio", lat_control)
            
            # Longitudinal control (elevator)
            self.send_dref("sim/cockpit2/controls/yoke_pitch_ratio", lon_control)
            
            # Rudder control
            self.send_dref("sim/cockpit2/controls/yoke_heading_ratio", rudder_control)
            
            # Throttle
            self.send_dref("sim/cockpit2/engine/actuators/throttle_jet_rev_ratio_all", throttle)
            
            # Landing gear
            self.send_dref("sim/cockpit/switches/gear_handle_status", gear)
            
            # Flaps
            self.send_dref("sim/cockpit2/controls/flap_ratio", flaps)
            
            # Speed brakes
            self.send_dref("sim/cockpit2/controls/speedbrake_ratio", speedbrakes)
            
            # Parking brake
            self.send_dref("sim/cockpit2/controls/parking_brake_ratio", park_brake)
        except Exception as e:
            print(f"Error sending control command: {e}")
    
    def pause_sim(self, pause: bool = True) -> None:
        """Pause or continue simulator
        
        Args:
            pause: True to pause, False to continue
        """
        command = 'sim/operation/pause_on' if pause else 'sim/operation/pause_off'
        self.send_command(command)
    
    def get_attitude(self) -> Dict[str, float]:
        """Get aircraft attitude data
        
        Returns:
            Dictionary containing attitude data:
            - roll: Roll angle (degrees)
            - pitch: Pitch angle (degrees)
            - heading: Heading (degrees)
            - alpha: Angle of attack (degrees)
            - beta: Sideslip angle (degrees)
        """
        try:
            roll = self.get_dref("sim/flightmodel/position/phi")
            pitch = self.get_dref("sim/flightmodel/position/theta")
            heading = self.get_dref("sim/flightmodel/position/psi")
            alpha = self.get_dref("sim/flightmodel/position/alpha")
            beta = self.get_dref("sim/flightmodel/position/beta")
            
            return {
                "roll": roll,
                "pitch": pitch, 
                "heading": heading,
                "alpha": alpha,
                "beta": beta
            }
        except Exception as e:
            print(f"Error getting attitude data: {e}")
            return {"roll": 0.0, "pitch": 0.0, "heading": 0.0, "alpha": 0.0, "beta": 0.0}
    
    def check_crash(self) -> bool:
        """Check if aircraft has crashed
        
        Returns:
            True if aircraft has crashed, False otherwise
        """
        try:
            crash = self.get_dref("sim/flightmodel2/misc/has_crashed")
            return crash > 0.5
        except Exception:
            return False 