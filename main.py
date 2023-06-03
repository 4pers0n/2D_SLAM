import env, sensors, feature
import pygame
import random
import math


def random_color():
    levels = range(32, 256, 32)
    return tuple(random.choice(levels) for _ in range(3))


featureMap = feature.FeaturesDetection()
environment = env.BuildEnvironment((600, 1200))
environment.originalMap = environment.map.copy()
laser = sensors.LaserSensor(200, environment.originalMap, uncertainty=(0.5, 0.01))
environment.map.fill((255, 255, 255))
environment.infomap = environment.map.copy()
originalMap = environment.map.copy()

running = True
FEATURE_DETECTION = True
BREAK_POINT_IND = 0

while running:
    environment.infomap = originalMap.copy()
    FEATURE_DETECTION = True
    BREAK_POINT_IND = 0
    ENDPOINTS = [0, 0]
    sensorOn = False
    PREDICTED_POINTS_TODRAW = []
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    if pygame.mouse.get_focused():
        sensorOn = True
    elif not pygame.mouse.get_focused():
        sensorOn = False

    if sensorOn:
        position = pygame.mouse.get_pos()
        laser.position = position
        sensor_data = laser.sense_obstacles()
        featureMap.laser_points_set(sensor_data)
        while BREAK_POINT_IND < (featureMap.NP - featureMap.PMIN):
            seedSeg = featureMap.seed_segment_detection(laser.position, BREAK_POINT_IND)
            if seedSeg == False:
                break
            else:
                seedSegment = seedSeg[0]
                PREDICTED_POINTS_TODRAW = seedSeg[1]
                INDICES = seedSeg[2]
                results = featureMap.seed_segment_growing(INDICES, BREAK_POINT_IND)
                if results == False:
                    BREAK_POINT_IND = INDICES[1]
                    continue
                else:
                    line_eq = results[1]
                    m, c = results[5]
                    line_seg = results[0]
                    OUTERMOST = results[2]
                    BREAK_POINT_IND = results[3]

                    ENDPOINTS[0] = featureMap.projection_point2line(OUTERMOST[0], m, c)
                    ENDPOINTS[1] = featureMap.projection_point2line(OUTERMOST[1], m, c)
                    featureMap.FEATURES.append([[m, c], ENDPOINTS])
                    pygame.draw.line(environment.infomap, (0, 255, 0), ENDPOINTS[0], ENDPOINTS[1], 1)
                    environment.dataStorage(sensor_data)

                    featureMap.FEATURES = featureMap.lineFeats2point()
                    feature.landmark_association(featureMap.FEATURES)
        for landmark in feature.Landmarks:
            pygame.draw.line(environment.infomap, (0, 0, 255), landmark[1][0], landmark[1][1], 2)

    environment.map.blit(environment.infomap, (0, 0))
    pygame.display.update()
