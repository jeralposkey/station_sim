# PRT Simulator - Final Version with Stats
import simpy
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# --- Simulation Parameters ---
SIMULATION_TIME = 300
NUM_VEHICLES = 18
RIDER_ARRIVAL_RATE = 1.0 / 3.0

# --- Physical Layout Parameters ---
TRACK_LENGTH = 2000.0
NUM_STATIONS = 3
STATION_LOCATIONS = [500.0, 1000.0, 1500.0]
BERTHS_PER_STATION = 3
SIDING_LENGTH = 100.0

# --- Vehicle Performance Parameters ---
MAINLINE_SPEED = 15.0
MAX_ACCEL = 2.0
MAX_DECEL = 3.0
VEHICLE_LENGTH = 3.0
MIN_SEPARATION_TIME = 3.0

class Track:
    def __init__(self, env, length):
        self.env = env
        self.length = length
        self.vehicles = []
    def add_vehicle(self, vehicle): self.vehicles.append(vehicle); self.vehicles.sort(key=lambda v: v.position)
    def remove_vehicle(self, vehicle):
        if vehicle in self.vehicles: self.vehicles.remove(vehicle)
    def get_leader(self, vehicle):
        if vehicle not in self.vehicles or len(self.vehicles) < 2: return None
        my_index = self.vehicles.index(vehicle)
        leader_index = (my_index + 1) % len(self.vehicles)
        return self.vehicles[leader_index]
    def get_distance_to_leader(self, vehicle, leader):
        if not leader: return float('inf')
        distance = leader.position - vehicle.position
        if distance < 0: distance += self.length
        return distance - VEHICLE_LENGTH
    def find_traffic_around(self, position):
        if not self.vehicles: return None, None
        positions = np.array([v.position for v in self.vehicles])
        dist_ahead = (positions - position + self.length) % self.length
        dist_behind = (position - positions + self.length) % self.length
        leader_idx = np.argmin(dist_ahead)
        follower_idx = np.argmin(dist_behind)
        return self.vehicles[follower_idx], self.vehicles[leader_idx]

class SerialStation:
    def __init__(self, env, name, track_pos, num_berths):
        self.env = env
        self.name = name
        self.track_pos = track_pos
        self.num_berths = num_berths
        self.berths = [None] * num_berths
        self.waiting_riders = []
        self.entry_point = (track_pos - (SIDING_LENGTH / 2)) % TRACK_LENGTH
        self.exit_point = (track_pos + (SIDING_LENGTH / 2)) % TRACK_LENGTH
    def is_full(self): return all(v is not None for v in self.berths)
    def add_vehicle(self, vehicle):
        for i in range(self.num_berths - 1, -1, -1):
            if self.berths[i] is None: self.berths[i] = vehicle; return True
        return False
    def depart_front_vehicle(self):
        if self.berths[0] is None: return
        departing_vehicle = self.berths[0]
        for i in range(self.num_berths - 1): self.berths[i] = self.berths[i+1]
        self.berths[-1] = None
        departing_vehicle.depart_from_station()

class Rider:
    def __init__(self, env, name, origin, destination):
        self.env = env
        self.name = name
        self.origin = origin
        self.destination = destination
        self.arrival_time = env.now
        self.pickup_time = -1
        self.dropoff_time = -1

class Vehicle:
    def __init__(self, env, name, prt_system):
        self.env = env
        self.name = name
        self.prt_system = prt_system
        self.position = 0.0
        self.velocity = 0.0
        self.state = 'on_mainline'
        self.destination_station = None
        self.current_station = None
        self.rider = None
        self.wake_up_event = env.event()
        self.action = env.process(self.run())

    def distance_to(self, target_pos):
        return (target_pos - self.position + self.prt_system.track.length) % self.prt_system.track.length

    def depart_from_station(self):
        self.state = 'merging'
        if not self.wake_up_event.triggered: self.wake_up_event.succeed()

    def run(self):
        dt = 0.2
        while True:
            if self.state == 'on_mainline':
                if self.destination_station:
                    dist_to_entry = self.distance_to(self.destination_station.entry_point)
                    required_decel_dist = (self.velocity**2) / (2 * MAX_DECEL) if MAX_DECEL > 0 else float('inf')
                    if dist_to_entry <= required_decel_dist: self.state = 'diverging'; continue
                leader = self.prt_system.track.get_leader(self)
                distance_to_leader = self.prt_system.track.get_distance_to_leader(self, leader)
                safe_dist = self.velocity * MIN_SEPARATION_TIME
                target_speed = MAINLINE_SPEED if distance_to_leader > safe_dist else (leader.velocity if leader else 0)
                if self.velocity < target_speed: self.velocity = min(target_speed, self.velocity + MAX_ACCEL * dt)
                elif self.velocity > target_speed: self.velocity = max(target_speed, self.velocity - MAX_DECEL * dt)
                self.position = (self.position + self.velocity * dt) % self.prt_system.track.length
                yield self.env.timeout(dt)
            elif self.state == 'diverging':
                entry_pos = self.destination_station.entry_point
                dist_to_entry = self.distance_to(entry_pos)
                if self.velocity * dt >= dist_to_entry and dist_to_entry > 0: self.position, self.velocity = entry_pos, 0
                else: self.velocity = max(0, self.velocity - MAX_DECEL * dt); self.position = (self.position + self.velocity * dt) % self.prt_system.track.length
                if self.velocity == 0:
                    self.prt_system.track.remove_vehicle(self)
                    self.destination_station.add_vehicle(self)
                    self.state = 'in_station'; self.current_station = self.destination_station; self.destination_station = None
                    if self.rider: self.rider.pickup_time = self.env.now
                    continue
                yield self.env.timeout(dt)
            elif self.state == 'merging':
                if self.rider: self.rider.dropoff_time = self.env.now; self.prt_system.add_completed_trip(self.rider); self.rider = None
                yield self.env.process(self.wait_for_gap_and_merge())
                self.state = 'on_mainline'; self.current_station = None
                continue
            elif self.state == 'in_station':
                yield self.wake_up_event
                self.wake_up_event = self.env.event()
                continue

    def wait_for_gap_and_merge(self):
        self.position = self.current_station.exit_point
        while True:
            follower, leader = self.prt_system.track.find_traffic_around(self.position)
            if not follower or not leader or follower == leader: break
            dist = self.prt_system.track.get_distance_to_leader(follower, leader)
            time_gap = dist / follower.velocity if follower.velocity > 0 else float('inf')
            time_needed = (MAINLINE_SPEED / MAX_ACCEL) + MIN_SEPARATION_TIME
            if time_gap >= time_needed: break
            yield self.env.timeout(0.5)
        self.prt_system.track.add_vehicle(self)

class PRTSystem:
    def __init__(self, env):
        self.env = env
        self.track = Track(env, TRACK_LENGTH)
        self.stations = [SerialStation(env, f"Station_{i}", loc, BERTHS_PER_STATION) for i, loc in enumerate(STATION_LOCATIONS)]
        self.vehicles = []
        self.history = []
        self.completed_trips = []
    def add_completed_trip(self, rider): self.completed_trips.append(rider)
    def vehicle_generator(self):
        for i in range(NUM_VEHICLES):
            v = Vehicle(self.env, f"Vehicle_{i}", self); v.position = (TRACK_LENGTH / NUM_VEHICLES) * i; v.velocity = MAINLINE_SPEED
            self.vehicles.append(v)
        self.track.vehicles = sorted(self.vehicles, key=lambda v: v.position)
    def rider_generator(self):
        rider_id = 0
        while True:
            yield self.env.timeout(random.expovariate(RIDER_ARRIVAL_RATE))
            rider_id += 1; origin, dest = random.sample(self.stations, 2)
            rider = Rider(self.env, f"Rider_{rider_id}", origin, dest)
            origin.waiting_riders.append(rider)
    def dispatcher(self):
        while True:
            for station in self.stations:
                # 1. Assign mainline vehicles to waiting riders
                if station.waiting_riders and not station.is_full():
                    available_vehicles = [v for v in self.vehicles if v.state == 'on_mainline' and v.destination_station is None]
                    if available_vehicles:
                        rider = station.waiting_riders.pop(0)
                        vehicle = min(available_vehicles, key=lambda v: v.distance_to(station.entry_point))
                        vehicle.destination_station = station; vehicle.rider = rider

                # 2. Handle boarding for any vehicle in any berth
                for vehicle in station.berths:
                    if vehicle and vehicle.rider and vehicle.state == 'in_station' and vehicle.destination_station is None:
                        vehicle.destination_station = vehicle.rider.destination

                # 3. Manage departures from the front berth
                front_vehicle = station.berths[0]
                if front_vehicle and front_vehicle.destination_station:
                    station.depart_front_vehicle()

            yield self.env.timeout(1)
    def record_snapshot(self, interval=1.0):
        while True:
            snapshot = {'time': self.env.now, 'vehicles': [{'name': v.name, 'pos': v.position, 'state': v.state, 'station_name': v.current_station.name if v.current_station else None} for v in self.vehicles], 'stations': {s.name: [v.name if v else None for v in s.berths] for s in self.stations}}; self.history.append(snapshot); yield self.env.timeout(interval)
    def print_statistics(self):
        print("\n--- Simulation Statistics ---")
        if not self.completed_trips: print("No trips were completed."); return
        print(f"Total completed trips: {len(self.completed_trips)}")
        wait_times = [(trip.pickup_time - trip.arrival_time) for trip in self.completed_trips if trip.pickup_time != -1]
        avg_wait = sum(wait_times)/len(wait_times) if wait_times else 0
        print(f"Average rider wait time: {avg_wait:.2f} seconds")
        trip_times = [(trip.dropoff_time - trip.pickup_time) for trip in self.completed_trips if trip.dropoff_time != -1]
        avg_trip = sum(trip_times)/len(trip_times) if trip_times else 0
        print(f"Average rider trip time: {avg_trip:.2f} seconds")

class Visualizer:
    def __init__(self, prt_system): self.prt_system = prt_system; self.fig, self.ax = plt.subplots(figsize=(15, 4)); self.vehicle_artists = {}; self.station_berth_texts = {}
    def setup_plot(self): self.ax.set_ylim(-2, 2); self.ax.set_xlim(-50, TRACK_LENGTH + 50); self.ax.set_yticks([]); self.ax.set_xlabel("Track Position (m)"); self.ax.plot([0, TRACK_LENGTH], [0, 0], 'k-', lw=3, alpha=0.5); [self.ax.plot([s.entry_point, s.exit_point], [0.5, 0.5], 'b-', lw=2) for s in self.prt_system.stations]; [self.ax.plot([s.entry_point, s.entry_point], [0, 0.5], 'b--', lw=1) for s in self.prt_system.stations]; [self.ax.plot([s.exit_point, s.exit_point], [0, 0.5], 'b--', lw=1) for s in self.prt_system.stations]; [self.ax.text(s.track_pos, 1.0, s.name, ha='center', weight='bold') for s in self.prt_system.stations]; self.station_berth_texts = {s.name: self.ax.text(s.track_pos, 0.7, "", ha='center', fontsize=8) for s in self.prt_system.stations}
    def animate(self, i):
        snapshot = self.prt_system.history[i]; self.ax.set_title(f"Personal Rapid Transit (PRT) Simulation | Time: {snapshot['time']:.2f}s")
        for s_name, berths in snapshot['stations'].items(): self.station_berth_texts[s_name].set_text(f"Berths: [ {' | '.join([v.replace('Vehicle_','V') if v else 'E' for v in berths])} ]")
        vehicle_data = {v['name']: v for v in snapshot['vehicles']}; current_vehicles_on_plot = set(vehicle_data.keys())
        for name, artist in list(self.vehicle_artists.items()):
            if name not in current_vehicles_on_plot: artist.remove(); del self.vehicle_artists[name]
        for name, v_data in vehicle_data.items():
            state, pos, station_name = v_data['state'], v_data['pos'], v_data['station_name']; y_pos = 0.5 if state == 'in_station' else 0
            if state == 'in_station' and station_name:
                station = next((s for s in self.prt_system.stations if s.name == station_name), None)
                if station:
                    try:
                        vehicle_obj = next(v for v in self.prt_system.vehicles if v.name == name)
                        berth_idx = station.berths.index(vehicle_obj)
                        pos = station.entry_point + (SIDING_LENGTH / (station.num_berths + 1)) * (station.num_berths - berth_idx)
                    except (ValueError, StopIteration): pass
            color_map = {'on_mainline': '#2ca02c', 'diverging': '#ff7f0e', 'in_station': '#1f77b4', 'merging': '#d62728'}
            if name not in self.vehicle_artists: self.vehicle_artists[name] = self.ax.plot(pos, y_pos, 'o', markersize=9)[0]
            self.vehicle_artists[name].set_data(pos, y_pos); self.vehicle_artists[name].set_color(color_map.get(state, 'black'))
        return self.vehicle_artists.values()
    def run_animation(self): self.setup_plot(); ani = animation.FuncAnimation(self.fig, self.animate, frames=len(self.prt_system.history), interval=100, blit=False, repeat=False); plt.tight_layout(); plt.show()

if __name__ == '__main__':
    print("--- PRT Full Simulation Starting ---")
    random.seed(42); env = simpy.Environment(); prt_system = PRTSystem(env)
    prt_system.vehicle_generator()
    env.process(prt_system.rider_generator())
    env.process(prt_system.dispatcher())
    env.process(prt_system.record_snapshot())
    env.run(until=SIMULATION_TIME)
    print(f"\n--- Simulation Finished ---")
    prt_system.print_statistics()
    print("\n--- Starting Visualization ---")
    vis = Visualizer(prt_system)
    vis.run_animation()
