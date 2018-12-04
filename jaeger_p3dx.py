import jaeger
import numpy as np
from gym import Env, spaces, envs
import skimage.color
import skimage.transform
import skimage.exposure
import random
import time

from image import find_object
from avoidObstacle import avoidObstacle_fuzzy

# [-180,180] angle: 0 -> 180 -> -180 -> 0
# [0,360] angle: 0 -> 360 -> 0
# [-360,360] angle: -360 -> 0 -> 360

# [-180,180] angle notation to [0,360]
def angle_notation_to_360(angle):
    if angle < 0:
        angle = np.deg2rad(360) + angle
    return angle

# [0,360] angle notation to [-180,180]
# [-360,360] angle notation to [-180,180]
def angle_notation_to_180(angle):
    if angle > np.deg2rad(180):
        angle = angle - np.deg2rad(360)
    elif angle < np.deg2rad(-180):
        angle = np.deg2rad(360) + angle
    return angle

#funcao de reward
def frame_step(angle_target, angle_atual, distance_target, distance_atual, frame):

    rate_angle = 1.2
    rate_distance = 1.0
    terminal = False
    
    error_angle = abs(angle_target - angle_atual)
    error_distance = abs(distance_target - distance_atual)
   
    if ((error_angle < 0.05)):
        reward_angle = rate_angle
    else:
        reward_angle = 0.916*(error_angle * (-rate_angle)) #maximo (error_angle) de foi definido como 0.5 sendo convertido para 0.55, caso não ocorra isso
        if (reward_angle < (-0.55)):
            reward_angle = -0.55

    if ((error_distance < 0.05)):
        reward_distance = rate_distance
    else:
        reward_distance = 0.4*(error_distance * (-rate_distance)) #maximo (error_distance) de foi definido como 1 sendo convertido para 0.4, caso não ocorra isso
        if (reward_distance < (-0.4)):
            reward_distance = -0.4

    x, y, ret = find_object(frame)

    #print('error angle '+str(error_angle))

    if (x == None or distance_atual<0.1 or error_distance>1.0 or error_angle>0.5):
        terminal = True
        reward_terminal = -1.0

    else:
        reward_terminal = 1.0

    reward_final = reward_angle + reward_distance + reward_terminal

    return reward_final, terminal


class P3DX_Env(Env):
    envs.registration.register(id="P3DX-v0",
        entry_point="jaeger_p3dx:P3DX_Env")

    def __init__(self):
        print("Creating env")
        self.interface = jaeger.Interface(interface="Vrep35", load=True)
        self.interface.connect()
        self.interface.load_scene("p3dxSimple.ttt")

        self.observation_space = spaces.Box(0.0, 1.0,
            shape=(80,80,4),
            dtype='float32')

        self.r2 = P3DX_Red("Pioneer_p3dx#0", self.interface)
        self.r1 = P3DX("Pioneer_p3dx", self.interface, self.r2)

    def step(self, action):
        self.r2.random_step()
        return self.r1.step(action)

    def reset(self):
        return self.r1.reset()


class P3DX(jaeger.Robot):

    def __init__(self, name, interface, second_robot):
        print("Creating robot:", name)
        # Acquires a robot instance from Jaeger/Vrep
        super().__init__(name, interface=interface)
        # Constants
        self.wheels_distance = 0.36205
        self.wheel_radius = 0.0975
        self.MAX_STEPS = 5000
        # needed to get the position of the other robot
        self.second_robot = second_robot
        # State: stack of 4 observations
        x_t = np.zeros((80,80),dtype='float')
        self.s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        # In Keras, need to reshape
        self.s_t = self.s_t.reshape(1, self.s_t.shape[0], self.s_t.shape[1], self.s_t.shape[2])  #1*80*80*4
        self.first_step = True

    def _move(self, left: float, right: float):
        self.Pioneer_p3dx_rightMotor.velocity = right
        self.Pioneer_p3dx_leftMotor.velocity = left

    def rotate_right(self, speed=2.0):
        self._move(speed, -speed)

    def rotate_left(self, speed=2.0):
        self._move(-speed, speed)

    def move_forward(self, speed=2.0):
        self._move(speed, speed)

    def move_backward(self, speed=2.0):
        self._move(-speed, -speed)

    def stop_moving(self):
        self._move(0, 0)

    def drive(self, linear_velocity, angular_velocity):
        left_velocity =  (2 * linear_velocity - self.wheels_distance * angular_velocity) / (2 * self.wheel_radius)
        right_velocity = (2 * linear_velocity + self.wheels_distance * angular_velocity) / (2 * self.wheel_radius)
        self._move(right_velocity, left_velocity)

    def actions(self, num):
        speed = 2.0
        if num == 0:
            self.stop_moving()
        elif num == 1:
            self.move_forward(speed)
        elif num == 2:
            self.rotate_left(speed)
        elif num == 3:
            self.rotate_right(speed)
        elif num == 4:
            self.move_backward(speed)
        elif num == 5:
            self.drive(speed/10, speed/10)
        elif num == 6:
            self.drive(speed/10, -speed/10)
        else:
            raise Exception("Action number not recognized")

    def get_vision(self):
        image_flat = self.Vision_sensor.reading
        if image_flat is None:
            return None
        image_flat = np.asarray(image_flat, dtype=np.uint8)
        image = image_flat.reshape((128, 128, 3))
        image = np.rot90(image, 2)
        return image

    def get_position(self):
        x, y, z = self.Pioneer_p3dx_visible.world_position
        alpha, beta, gamma = self.Pioneer_p3dx_visible.world_orientation[0]
        return x, y, gamma

    def distance_and_angle_from_second_robot(self):
        x_robot, y_robot, theta_robot = self.get_position()
        x_robot2, y_robot2, _ = self.second_robot.get_position()
        dist = np.sqrt((x_robot2 - x_robot)**2 + (y_robot2 - y_robot)**2)
        # 180 notation: angle of the vector from robot to destination
        dest_angle = np.arctan2(y_robot2 - y_robot, x_robot2 - x_robot)
        # 360 notation: angle of the vector from robot to destination
        dest_angle360 = angle_notation_to_360(dest_angle)
        # 360 notation: robot angle
        robot_angle360 = angle_notation_to_360(theta_robot)
        # 360 notation: angle difference
        angle_dist360 = dest_angle360  - robot_angle360
        # 180 notation: angle difference
        angle_dist = angle_notation_to_180(angle_dist360)
        return dist, angle_dist

    def step(self, num):
        #print("Step", self.interface.steps)

        # set targets
        if self.first_step:
            self.distance_target, self.angle_target = self.distance_and_angle_from_second_robot()
            while self.distance_target == 0.0 and self.angle_target == 0.0:
                self.distance_target, self.angle_target = self.distance_and_angle_from_second_robot()
            self.first_step = False
            time.sleep(0.0001)
            #print("targets:", self.distance_target, self.angle_target)

        # action
        self.actions(num)
        self.interface.step()

        def get_vision_wrapper():
            try:
                return self.get_vision()
            except:
                return None

        observation = None
        while observation is None:
            observation = get_vision_wrapper()

        dist, angle = self.distance_and_angle_from_second_robot()

        
        # reward
        reward_final, done = frame_step(self.angle_target, angle, self.distance_target, dist, observation)

        x_t = skimage.color.rgb2gray(observation)
        x_t = skimage.transform.resize(x_t,(80,80))
        x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))

        x_t = x_t / 255.0
        x_t = x_t.reshape(1, x_t.shape[0], x_t.shape[1], 1) #1x80x80x1
        self.s_t = np.append(x_t, self.s_t[:, :, :, :3], axis=3)

        # done if over max steps
        if not done and self.interface.steps > self.MAX_STEPS:
            done = True

        # return (observation, reward, done, info)
        return (self.s_t, reward_final, done, '')

    def reset(self):
        self.interface.reset()
        # State: stack of 4 observations
        x_t = np.zeros((80,80),dtype='float')
        self.s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        #In Keras, need to reshape
        self.s_t = self.s_t.reshape(1, self.s_t.shape[0], self.s_t.shape[1], self.s_t.shape[2])  #1*80*80*4
        return self.s_t


class P3DX_Red(jaeger.Robot):

    def __init__(self, name, interface):
        print("Creating robot:", name)
        # Acquires a robot instance from Jaeger/Vrep
        super().__init__(name, interface=interface)
        # Constants
        self.wheels_distance = 0.36205
        self.wheel_radius = 0.0975
        self.sonar_detect_min_dist = 0.2
        self.sonar_detect_max_dist = 0.5
        # Random movement
        self.iteration_with_nothing_on_sonar = 0
        self.random_linear = 0.0
        self.random_angular = 0.0
        self.select_new_random_movement = True

    def _move(self, left: float, right: float):
        self.Pioneer_p3dx_rightMotor.velocity = right
        self.Pioneer_p3dx_leftMotor.velocity = left

    def move_forward(self, speed=2.0):
        self._move(speed, speed)

    def drive(self, linear_velocity, angular_velocity):
        left_velocity =  (2 * linear_velocity - self.wheels_distance * angular_velocity) / (2 * self.wheel_radius)
        right_velocity = (2 * linear_velocity + self.wheels_distance * angular_velocity) / (2 * self.wheel_radius)
        self._move(right_velocity, left_velocity)

    def get_position(self):
        x, y, z = self.Pioneer_p3dx_visible.world_position
        alpha, beta, gamma = self.Pioneer_p3dx_visible.world_orientation[0]
        return x, y, gamma

    def fix_sonar_reading(self, reading):
        if reading == -1 or reading > self.sonar_detect_max_dist:
            reading = self.sonar_detect_max_dist
        # If reading is closer than minimum distance
        if reading < self.sonar_detect_min_dist:
            reading = self.sonar_detect_min_dist
        return reading

    def random_step(self):
        def get_random_movement():
            number = random.randint(1,10)
            linear = angular = 0.0
            # 40% of chance
            if number <= 4:
                # move forward
                linear = 0.2
                angular = 0.0
            # 60% of chance
            else:
                # random movement
                linear = round(random.uniform(0.05, 0.30), 2)
                angular = round(random.uniform(-0.30, 0.30), 2)
            return linear, angular

        def get_sensor_data():
            s1 = None
            s2 = None
            s3 = None
            s4 = None
            try:
                r1 = self.Pioneer_p3dx_ultrasonicSensor2.reading
                if r1[0]:
                    s1 = r1[1][2]
                else:
                    s1 = self.sonar_detect_max_dist
                r2 = self.Pioneer_p3dx_ultrasonicSensor4.reading
                if r2[0]:
                    s2 = r2[1][2]
                else:
                    s2 = self.sonar_detect_max_dist
                r3 = self.Pioneer_p3dx_ultrasonicSensor5.reading
                if r3[0]:
                    s3 = r3[1][2]
                else:
                    s3 = self.sonar_detect_max_dist
                r4 = self.Pioneer_p3dx_ultrasonicSensor7.reading
                if r4[0]:
                    s4 = r4[1][2]
                else:
                    s4 = self.sonar_detect_max_dist
                return [s1, s2, s3, s4]
            except:
                return [s1, s2, s3, s4]

        l = [None, None, None, None]
        while(l[0] is None or l[1] is None or l[2] is None or l[3] is None):
            l = get_sensor_data()

        s1 = self.fix_sonar_reading(l[0])
        s2 = self.fix_sonar_reading(l[1])
        s3 = self.fix_sonar_reading(l[2])
        s4 = self.fix_sonar_reading(l[3])

        # avoid obstacle if there is a detection
        if s1 < 0.5 or s2 < 0.5 or s3 < 0.5 or s4 < 0.5:
            #print('Avoiding obstacle', s1, s2, s3, s4)
            linear, angular = avoidObstacle_fuzzy(s1, s2, s3, s4)
            self.drive(linear, angular)
            self.select_new_random_movement = True
            self.iteration_with_nothing_on_sonar = 0
        # move randomly
        else:
            # move forward for a few iterations
            if self.iteration_with_nothing_on_sonar < 5:
                #print('Moving forward first')
                self.move_forward()
                self.iteration_with_nothing_on_sonar += 1
            # then, random movement
            else:
                if self.select_new_random_movement:
                    self.random_linear, self.random_angular = get_random_movement()
                    #print('selected random movement', self.random_linear, self.random_angular)
                    self.select_new_random_movement = False
                self.drive(self.random_linear, self.random_angular)
                self.iteration_with_nothing_on_sonar += 1

        # selects new random movement if it stays in one for too long
        if self.iteration_with_nothing_on_sonar != 0 and self.iteration_with_nothing_on_sonar % 20 == 0:
            self.random_linear, self.random_angular = get_random_movement()
            #print('switch selected random movement', self.random_linear, self.random_angular)
