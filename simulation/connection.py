import os
import sys
import glob

try:
    sys.path.append(glob.glob('./carla/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    print(glob.glob('./carla/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print('Couldn\'t import Carla egg properly')

import carla
# print(dir(carla))
from simulation.settings import PORT, TIMEOUT, HOST

class ClientConnection:
    def __init__(self, town):
        self.client = None
        self.town = town

    def setup(self):
        try:

            # Connecting to the  Server
            self.client = carla.Client(HOST, PORT)
            self.client.set_timeout(TIMEOUT)
            
            # Normalize town name (CARLA expects exact case like "Town07", "Town02", etc.)
            town_name = self.town
            
            # Get available maps to validate
            available_maps = self.client.get_available_maps()
            available_town_names = [m.split('/')[-1] for m in available_maps]
            print(f"Available towns: {available_town_names}")
            print(f"Requested town: {town_name}")
            
            # Try to find matching town name (case-insensitive)
            matching_town = None
            for available_town in available_town_names:
                if available_town.lower() == town_name.lower():
                    matching_town = available_town
                    break
            
            if matching_town:
                town_name = matching_town
                print(f"Using town: {town_name}")
            else:
                print(f"Warning: Town '{town_name}' not found in available maps. Trying anyway...")
            
            self.world = self.client.load_world(town_name)
            self.world.set_weather(carla.WeatherParameters.CloudyNoon)
            return self.client, self.world

        except Exception as e:
            print(
                'Failed to make a connection with the server: {}'.format(e))
            if self.client:
                self.error()
            raise  # Re-raise the exception so caller can handle it

    # An error method: prints out the details if the client failed to make a connection
    def error(self):

        print("\nClient version: {}".format(
            self.client.get_client_version()))
        print("Server version: {}\n".format(
            self.client.get_server_version()))

        if self.client.get_client_version != self.client.get_server_version:
            print(
                "There is a Client and Server version mismatch! Please install or download the right versions.")
