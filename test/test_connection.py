"""
Test XPlane Connection Functionality

This script tests if the UDP connection to the XPlane simulator is working properly,
and tests the basic functionality of subscribing to DataRefs and getting data.
"""

import sys
import os
import time

# Add project path to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from XPlaneGym.xplane_connect import XPlaneConnect

def test_connection(ip='127.0.0.1', port=49000, timeout=1.0):
    """Test connection to XPlane and try to get basic data"""
    print(f"Testing connection to XPlane: {ip}:{port}")
    
    # Create connection
    try:
        client = XPlaneConnect(ip=ip, port=port, timeout=timeout)
        print("✓ Connection created successfully")
    except Exception as e:
        print(f"✗ Connection creation failed: {e}")
        return False
    
    # Test subscribing to DataRefs
    try:
        drefs = [
            ("sim/flightmodel/position/latitude", 10),
            ("sim/flightmodel/position/longitude", 10),
            ("sim/flightmodel/position/elevation", 10)
        ]
        client.subscribe_drefs(drefs)
        print("✓ DataRefs subscription successful")
    except Exception as e:
        print(f"✗ DataRefs subscription failed: {e}")
        return False
    
    # Wait for data reception
    print("Waiting for data reception...")
    time.sleep(2.0)
    
    # Try to get position data
    try:
        position = client.get_position()
        if position and len(position) >= 3:
            lat, lon, alt = position[0:3]
            print(f"✓ Position data retrieved successfully:")
            print(f"  - Latitude: {lat:.6f}°")
            print(f"  - Longitude: {lon:.6f}°")
            print(f"  - Altitude: {alt:.2f} meters")

        else:
            print("✗ Position data incomplete")
            return False
    except Exception as e:
        print(f"✗ Position data retrieval failed: {e}")
        return False
    
    # Try to get attitude data
    try:
        attitude = client.get_attitude()
        print(f"✓ Attitude data retrieved successfully:")
        print(f"  - Roll: {attitude['roll']:.2f}°")
        print(f"  - Pitch: {attitude['pitch']:.2f}°")
        print(f"  - Heading: {attitude['heading']:.2f}°")
    except Exception as e:
        print(f"✗ Attitude data retrieval failed: {e}")
        return False
    
    # Close connection
    try:
        client.stop()
        print("✓ Connection closed successfully")
    except Exception as e:
        print(f"✗ Connection closure failed: {e}")
        return False
    
    print("All connection tests passed!")
    return True

if __name__ == "__main__":
    # Get IP and port from command line arguments (if provided)
    ip = sys.argv[1] if len(sys.argv) > 1 else '127.0.0.1'
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 49000
    
    test_connection(ip, port) 