# PRT Simulator
import simpy
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Simulation Parameters ---
SIMULATION_TIME = 200 # Shorter time for quicker visualization run
RIDER_ARRIVAL_RATE = 1.0 / 4
NUM_STATIONS = 3
STATION_BERTHS = [2, 3, 2]
NUM_VEHICLES = 15

# --- Vehicle Performance Parameters ---
MAINLINE_SPEED = 15.0  # m/s
MAX_ACCEL = 2.0  # m/s^2
MAX_DECEL = 3.0  # m/s^2

# --- Physical Layout Parameters ---
TRACK_LENGTH = 1000.0
STATION_LOCATIONS = [200.0, 500.0, 800.0]

class Station:
    """Represents a station with berths and a queue of waiting riders."""
    def __init__(self, env, name, num_berths, position):
        self.env = env
        self.name = name
        self.num_berths = num_berths
        self.berths = simpy.Resource(env, capacity=num_berths)
        self.position = position
        self.waiting_riders = []
        self.total_busy_time = 0.0

class Rider:
    """Represents a passenger with an origin and a destination."""
    def __init__(self, env, name, origin, destination):
        self.env = env
        self.name = name
        self.origin = origin
        self.destination = destination
        self.arrival_time = env.now
        self.pickup_time = -1
        self.dropoff_time = -1
        # print(f"Time {env.now:.2f}: {name} arrives at {origin.name}, wants to go to {destination.name}.")

class Vehicle:
    """A vehicle that can be idle, pick up a rider, and drop them off."""
    def __init__(self, env, name, prt_system, initial_position=0.0):
        self.env = env
        self.name = name
        self.prt_system = prt_system
        self.position = initial_position
        self.velocity = 0.0
        self.state = 'idle'
        self.trip = None
        self.action = env.process(self.run())

    def assign_trip(self, rider):
        self.trip = rider
        self.state = 'en_route_to_pickup'
        # print(f"Time {self.env.now:.2f}: {self.name} assigned to {rider.name} at {rider.origin.name}.")

    def run(self):
        while True:
            if self.state == 'idle':
                yield self.env.timeout(1)
                continue

            if self.state == 'en_route_to_pickup':
                station = self.trip.origin
                yield self.env.process(self.drive_to(station.position))
                self.state = 'dwelling_pickup'

            elif self.state == 'dwelling_pickup':
                station = self.trip.origin
                with station.berths.request() as req:
                    yield req
                    start_dwell = self.env.now
                    # print(f"Time {self.env.now:.2f}: {self.name} docked at {station.name} for pickup.")
                    yield self.env.timeout(10)
                    self.trip.pickup_time = self.env.now
                    station.total_busy_time += (self.env.now - start_dwell)
                    # print(f"Time {self.env.now:.2f}: {self.name} picked up {self.trip.name}.")
                self.state = 'en_route_to_dropoff'

            elif self.state == 'en_route_to_dropoff':
                station = self.trip.destination
                yield self.env.process(self.drive_to(station.position))
                self.state = 'dwelling_dropoff'

            elif self.state == 'dwelling_dropoff':
                station = self.trip.destination
                with station.berths.request() as req:
                    yield req
                    start_dwell = self.env.now
                    # print(f"Time {self.env.now:.2f}: {self.name} docked at {station.name} for dropoff.")
                    yield self.env.timeout(5)
                    self.trip.dropoff_time = self.env.now
                    station.total_busy_time += (self.env.now - start_dwell)
                    self.prt_system.add_completed_trip(self.trip)
                    # print(f"Time {self.env.now:.2f}: {self.name} dropped off {self.trip.name}.")
                self.trip = None
                self.state = 'idle'
                # print(f"Time {self.env.now:.2f}: {self.name} is now idle at {station.name}.")

    def drive_to(self, destination):
        dt = 0.2
        while abs(self.position - destination) > 1.0:
            dist_to_dest = abs(self.position - destination)
            required_decel_dist = (self.velocity**2) / (2 * MAX_DECEL) if MAX_DECEL > 0 else float('inf')

            if dist_to_dest <= required_decel_dist:
                self.velocity = max(0, self.velocity - MAX_DECEL * dt)
            else:
                if self.velocity < MAINLINE_SPEED:
                    self.velocity = min(MAINLINE_SPEED, self.velocity + MAX_ACCEL * dt)

            direction = 1 if destination > self.position else -1
            self.position += self.velocity * direction * dt
            yield self.env.timeout(dt)
        self.position = destination
        self.velocity = 0.0

class PRTSystem:
    """Manages the PRT simulation environment, including a dispatcher."""
    def __init__(self, env):
        self.env = env
        self.stations = [Station(env, f"Station_{i}", STATION_BERTHS[i], loc) for i, loc in enumerate(STATION_LOCATIONS)]
        self.vehicles = []
        self.completed_trips = []
        self.history = []

    def add_completed_trip(self, rider):
        self.completed_trips.append(rider)

    def record_snapshot(self, interval=1.0):
        """Records snapshots of the system state for visualization."""
        while True:
            snapshot = {
                'time': self.env.now,
                'vehicles': [{'name': v.name, 'pos': v.position, 'state': v.state} for v in self.vehicles]
            }
            self.history.append(snapshot)
            yield self.env.timeout(interval)

    def vehicle_generator(self, num_vehicles):
        for i in range(num_vehicles):
            pos = self.stations[0].position
            v = Vehicle(self.env, f"Vehicle_{i}", self, initial_position=pos)
            self.vehicles.append(v)
            yield self.env.timeout(0.1)

    def rider_generator(self):
        rider_id = 0
        while True:
            yield self.env.timeout(random.expovariate(RIDER_ARRIVAL_RATE))
            rider_id += 1
            origin, dest = random.sample(self.stations, 2)
            Rider(self.env, f"Rider_{rider_id}", origin, dest)
            origin.waiting_riders.append(Rider(self.env, f"Rider_{rider_id}", origin, dest))

    def dispatcher(self):
        while True:
            idle_vehicles = [v for v in self.vehicles if v.state == 'idle']
            if idle_vehicles:
                for station in self.stations:
                    if station.waiting_riders:
                        vehicle = self.find_closest_idle_vehicle(station, idle_vehicles)
                        if vehicle:
                            rider = station.waiting_riders.pop(0)
                            vehicle.assign_trip(rider)
                            idle_vehicles.remove(vehicle)
                            if not idle_vehicles:
                                break
            yield self.env.timeout(1)

    def find_closest_idle_vehicle(self, station, idle_vehicles):
        if not idle_vehicles: return None
        return min(idle_vehicles, key=lambda v: abs(v.position - station.position))

    def print_statistics(self):
        print("\n--- Simulation Statistics ---")
        if not self.completed_trips:
            print("No trips were completed.")
            return
        print(f"Total completed trips (throughput): {len(self.completed_trips)}")
        wait_times = [(trip.pickup_time - trip.arrival_time) for trip in self.completed_trips if trip.pickup_time != -1]
        avg_wait_time = sum(wait_times) / len(wait_times) if wait_times else 0
        max_wait_time = max(wait_times) if wait_times else 0
        print(f"Average rider wait time: {avg_wait_time:.2f} seconds")
        print(f"Maximum rider wait time: {max_wait_time:.2f} seconds")
        trip_times = [(trip.dropoff_time - trip.pickup_time) for trip in self.completed_trips if trip.dropoff_time != -1]
        avg_trip_time = sum(trip_times) / len(trip_times) if trip_times else 0
        max_trip_time = max(trip_times) if trip_times else 0
        print(f"Average rider trip time: {avg_trip_time:.2f} seconds")
        print(f"Maximum rider trip time: {max_trip_time:.2f} seconds")
        print("Berth Utilization:")
        for station in self.stations:
            if station.num_berths > 0 and SIMULATION_TIME > 0:
                utilization = (station.total_busy_time / (station.num_berths * SIMULATION_TIME)) * 100
                print(f"  - {station.name}: {utilization:.2f}%")

class Visualizer:
    """Creates an animation of the PRT system from recorded history."""
    def __init__(self, prt_system):
        self.prt_system = prt_system
        self.fig, self.ax = plt.subplots(figsize=(15, 3))
        self.vehicle_artists = {}

    def setup_plot(self):
        self.ax.set_ylim(-2, 2)
        self.ax.set_xlim(-50, TRACK_LENGTH + 50)
        self.ax.set_yticks([])
        self.ax.set_xlabel("Track Position (m)")
        self.ax.plot([0, TRACK_LENGTH], [0, 0], 'k-', lw=2)
        for station in self.prt_system.stations:
            self.ax.plot(station.position, 0, 's', markersize=12, color='blue')
            self.ax.text(station.position, 0.2, station.name, ha='center', fontsize=9)

    def animate(self, i):
        snapshot = self.prt_system.history[i]
        time = snapshot['time']
        self.ax.set_title(f"PRT System Simulation | Time: {time:.2f}s")

        current_vehicles_on_plot = {v['name'] for v in snapshot['vehicles']}

        # Update and remove
        for name, artist in list(self.vehicle_artists.items()):
            if name not in current_vehicles_on_plot:
                artist.remove()
                del self.vehicle_artists[name]

        # Add and update
        for v in snapshot['vehicles']:
            name, pos, state = v['name'], v['pos'], v['state']
            color_map = {'idle': 'gray', 'en_route_to_pickup': 'orange', 'dwelling_pickup': 'red', 'en_route_to_dropoff': 'green', 'dwelling_dropoff': 'purple'}
            color = color_map.get(state, 'black')

            if name in self.vehicle_artists:
                self.vehicle_artists[name].set_data(pos, 0)
                self.vehicle_artists[name].set_color(color)
            else:
                line, = self.ax.plot(pos, 0, 'o', markersize=8, color=color)
                self.vehicle_artists[name] = line

        return self.vehicle_artists.values()

    def run_animation(self):
        self.setup_plot()
        ani = animation.FuncAnimation(self.fig, self.animate, frames=len(self.prt_system.history), interval=100, blit=False, repeat=False)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    print("--- PRT Simulation Starting ---")
    random.seed(42)
    env = simpy.Environment()
    prt_system = PRTSystem(env)

    env.process(prt_system.vehicle_generator(NUM_VEHICLES))
    env.process(prt_system.rider_generator())
    env.process(prt_system.dispatcher())
    env.process(prt_system.record_snapshot())

    env.run(until=SIMULATION_TIME)
    print("\n--- PRT Simulation Finished ---")
    prt_system.print_statistics()

    print("\n--- Starting Visualization ---")
    vis = Visualizer(prt_system)
    vis.run_animation()
