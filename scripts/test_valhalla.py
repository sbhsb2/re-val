"""
Test Valhalla Connection
=========================

Simple script to verify Valhalla routing engine is accessible.
Tests basic routing between two points in Bremen.

Usage: python test_valhalla.py
"""

import requests
import json

VALHALLA_URL = "http://localhost:8002"

def test_valhalla_status():
    """
    BLOCK 1: Test if Valhalla server is running
    
    What it does:
    - Sends GET request to /status endpoint
    - Checks if server responds
    
    Why: Need to verify Valhalla is accessible before running analysis
    """
    print("="*70)
    print("TESTING VALHALLA CONNECTION")
    print("="*70)
    
    try:
        response = requests.get(f"{VALHALLA_URL}/status", timeout=5)
        if response.status_code == 200:
            print("✓ Valhalla is running!")
            print(f"  Response: {response.text[:100]}...")
            return True
        else:
            print(f"✗ Valhalla returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to Valhalla!")
        print("  Make sure Valhalla is running:")
        print("  docker start valhalla-service")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_routing():
    """
    BLOCK 2: Test basic routing calculation
    
    What it does:
    - Requests route between two points in Bremen
    - Point 1: Bremen city center (53.0793°N, 8.8017°E)
    - Point 2: Bremen University (53.0758°N, 8.8072°E)
    
    Returns:
    - Distance in kilometers
    - Travel time in seconds
    
    Why: Verify routing engine works with Bremen data
    """
    print("\n" + "="*70)
    print("TESTING BASIC ROUTING")
    print("="*70)
    
    # Test coordinates in Bremen
    start_lat = 53.0793
    start_lon = 8.8017
    end_lat = 53.0758
    end_lon = 8.8072
    
    print(f"\nCalculating route:")
    print(f"  From: ({start_lat}, {start_lon})")
    print(f"  To:   ({end_lat}, {end_lon})")
    
    try:
        response = requests.post(
            f"{VALHALLA_URL}/route",
            json={
                'locations': [
                    {'lat': start_lat, 'lon': start_lon},
                    {'lat': end_lat, 'lon': end_lon}
                ],
                'costing': 'auto',  # Car routing
                'directions_options': {'units': 'kilometers'}
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            summary = result['trip']['summary']
            
            distance_km = summary['length']
            time_seconds = summary['time']
            time_minutes = time_seconds / 60
            speed_kmh = (distance_km / time_seconds) * 3600 if time_seconds > 0 else 0
            
            print("\n✓ Routing successful!")
            print(f"  Distance: {distance_km:.2f} km")
            print(f"  Time: {time_minutes:.2f} minutes ({time_seconds:.0f} seconds)")
            print(f"  Average speed: {speed_kmh:.1f} km/h")
            
            return {
                'distance_km': distance_km,
                'time_seconds': time_seconds,
                'speed_kmh': speed_kmh
            }
        else:
            print(f"\n✗ Routing failed with status code: {response.status_code}")
            print(f"  Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"\n✗ Error during routing: {e}")
        return None


def main():
    """
    BLOCK 3: Main execution
    
    What it does:
    - Runs status check
    - If successful, runs routing test
    - Reports overall results
    
    Why: Ensures Valhalla is properly configured before running full analysis
    """
    print("\n" + "="*70)
    print("VALHALLA CONNECTIVITY TEST")
    print("="*70)
    
    # Test 1: Is Valhalla running?
    status_ok = test_valhalla_status()
    
    if not status_ok:
        print("\n" + "="*70)
        print("❌ TESTS FAILED")
        print("="*70)
        print("\nValhalla is not accessible. Please:")
        print("1. Make sure Docker is running")
        print("2. Start Valhalla: docker start valhalla-service")
        print("3. Wait 30 seconds for Valhalla to initialize")
        print("4. Run this test again")
        return
    
    # Test 2: Can we calculate routes?
    route_result = test_routing()
    
    if route_result:
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\nValhalla is ready for analysis!")
        print("You can now run the speed analysis script.")
    else:
        print("\n" + "="*70)
        print("⚠️ PARTIAL SUCCESS")
        print("="*70)
        print("\nValhalla is running but routing failed.")
        print("This might mean Bremen data is not loaded.")


if __name__ == '__main__':
    main()
