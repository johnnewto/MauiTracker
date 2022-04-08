
import pymavlink
import time
import math

from pymavlink import mavutil
print(pymavlink.__doc__)
source_system = 128



class ConnectQGC:
    def __init__(self, pos):
        self.master = mavutil.mavlink_connection('udpout:localhost:14550', source_system=source_system)
        self.lat = pos[0]
        self.lon = pos[1]
        self.adsb_objs = {}
        self.assign_ICAO = 9999000

    def statustext_send(self, txt):
        self.master.mav.statustext_send(mavutil.mavlink.MAV_SEVERITY_NOTICE,
                                   "QGC will read this".encode())

    def home_position_send(self):
        self.master.mav.home_position_send(
            int(self.lat * 1e7),  # 'latitself.adsb_objsude',
            int(self.lon * 1e7),  # 'longitude',
            int(1000),  # 'altitude',
            0.0,  # 'x',
            0.0,  # 'y',
            0.0,  # 'z',
            (100, 0, 0, 0),  # 'q',
            0.0,  # 'approach_x',
            0.0,  # 'approach_y',
            0.0  # 'approach_z'
        )

    def high_latency2_send(self, lat=None, lon=None, heading=None):
        if lat is not None:
            self.lat = lat
        if lon is not None:
            self.lon = lon
        if heading is not None:
            self.heading = int(heading) % 360

        self.master.mav.high_latency2_send(
                        0,    # timestamp
                        0,    # type
                        0,    # autopilot
                        0,    # custom_mode
                        int(self.lat * 1e7), # 'lat',
                        int(self.lon * 1e7), # 'lon',
                        int(10000), # 'alt',
                        0,    # target_altitude',
                        int(self.heading//2),    # heading
                        0,    # target_heading
                        0,    # target_distance
                        0,    # throttle
                        0,    # airspeed
                        0,    # airspeed_sp
                        0,    # groundspeed
                        0,    # windspeed
                        0,    # wind_heading
                        0,    # eph
                        0,    # epv
                        0,    # temperature_air
                        0,    # climb_rate
                        100,    # battery',
                        0,    # wp_num
                        0,    # failure_flags
                        0,    # custom0
                        0,    # custom1
                        0    # custom2'
        )
        # master.mav.sys_status_send(
        #    0,  # 'onboard_control_sensors_present',
        #    0,  # 'onboard_control_sensors_enabled',
        #    0,  # 'onboard_control_sensors_health',
        #    250,  # 'load',
        #    1680,  # 'voltage_battery',
        #    800,  # 'current_battery',
        #    99,  # 'battery_remaining',
        #    0,  # 'drop_rate_comm',
        #    0,  # 'errors_comm',
        #    0,  # 'errors_count1',
        #    0,  # 'errors_count2',
        #    0,  # 'errors_count3',
        #    0  # 'errors_count4'
        # )
        # master.mav.heartbeat_send(
        #     2,  # ype
        #     3,  # autopilot
        #     65,  # base_mode
        #     65536,  # custom_mode
        #     3,  # system_status
        #     3
        # )
    def adsb_vehicle_send(self, name, distance, ang, ICAO=None, max_step=50):
        try:
            self.last_adsb = self.adsb_objs[name]
            ICAO = self.adsb_objs[name].ICAO
        except:
            if ICAO is None:
                self.assign_ICAO += 1
                ICAO = self.assign_ICAO

            self.adsb_objs[name] = ADSB_object((self.lat, self.lon), max_step)
            self.adsb_objs[name].ICAO = ICAO
            self.last_adsb = self.adsb_objs[name]

        lat0, lon0 = self.adsb_objs[name].offset(distance, ang)
        self.master.mav.adsb_vehicle_send(
            ICAO, # ICAO address
            int(lat0* 1e7),
            int(lon0 * 1e7),
            1,
            int(10000), # 10m up
            1000, # heading in cdeg
            0, # horizontal velocity cm/s
            0, # vertical velocity cm/s
            name.encode("ascii"), # callsign
            7,
            1, # time since last communication
            87, # flags
            0 # squawk
        )

# Press the green button in the gutter to run the script.

class ADSB_object:
    def __init__(self, pos, max_step=50):
        self.master = mavutil.mavlink_connection('udpout:localhost:14550', source_system=source_system)
        self.lat = pos[0]
        self.lon = pos[1]
        self.max_step = max_step

    def offset(self, dist, ang):
        # Earth’s radius, sphere
        R = 6378137
        # offsets in meters
        dn = dist * math.cos(math.radians(ang))
        de = dist * math.sin(math.radians(ang))
        try:
            self.ddist = 0.0
            ddn = dn - self.last_dn
            dde = de - self.last_de
            ddist = math.sqrt(ddn*ddn + dde*dde)
            # ratio of distance to max distance step
            self.ratio = max(ddist/self.max_step, 1.0)
            self.dn = self.last_dn + ddn / self.ratio
            self.de = self.last_de + dde / self.ratio
            ddn = self.dn - self.last_dn
            dde = self.de - self.last_de
            self.ddist = math.sqrt(ddn * ddn + dde * dde)
            # print(int(self.last_dn), int(self.last_de))
            # print(int(self.dn), int(self.de))
            # print (ddn, dde, self.ddist)
            # assert self.ddist <= self.max_step

        except AttributeError:
            print("AttributeError")
            self.ratio = 1.0   # ratio is set to limit the max distance step
            self.dn = dn
            self.de = de

        self.last_dn = self.dn
        self.last_de = self.de

        # print(dist, ang, self.dn, self.de)

        # Coordinate offsets in radians
        dLat = self.dn / R
        dLon = self.de / (R * math.cos(math.pi * self.lat  / 180))

        # OffsetPosition, decimal degrees
        lat0 = self.lat + dLat * 180 / math.pi
        lon0 = self.lon + dLon * 180 / math.pi
        return lat0, lon0


if __name__ == '__main__':
    GERMAN_START_POS = (47.392, 8.542)
    NZ_START_POS = (-36.863898960619764, 174.84190246178986)
    NZ_START_POS = (-36.9957915731748, 174.91686500754628)
    qgc = ConnectQGC(NZ_START_POS)
    for i in range(10):
        (lat, lon) = NZ_START_POS
        time.sleep(0.5)
        print("message", i)
        ang = 0+10*i
        dist = 1000
        qgc.high_latency2_send(lat, lon, ang)

        qgc.adsb_vehicle_send('terry', dist, ang, max_step=50)
        qgc.adsb_vehicle_send('bob', dist*1.5, ang, max_step=50)
        print(int(qgc.last_adsb.dn), int(qgc.last_adsb.de), int(qgc.last_adsb.ddist))