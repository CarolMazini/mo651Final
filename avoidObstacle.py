import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

def avoidObstacle_fuzzy(s1Distance, s2Distance, s3Distance, s4Distance):
    # Generate universe variables

    #Sonar Typical range: 5 - 12cm to 5m
    #sonar_dtct_min_dist = 0.2
    #sonar_dtct_max_dist = 0.5 // farther obstacles are not detected
    x_distSonar = np.arange(0.19, 0.51, 0.01)
    x_angular = np.arange(-1.5, 1.6, 0.01)
    x_linear = np.arange(-0.3, 0.4, 0.01)

    # Generate fuzzy membership functions
    #input
    #distSonar_tooClose = fuzz.trimf(x_distSonar, [0.19, 0.19, 0.24])
    distSonar_tooClose = fuzz.trapmf(x_distSonar, [0, 0, 0.2, 0.33]) #0, 0, 0.2, 0.3
    distSonar_close = fuzz.trimf(x_distSonar, [0.25, 0.32, 0.36])#0.25, 0.3, 0.35
    distSonar_medium = fuzz.trimf(x_distSonar, [0.34, 0.37, 0.4])#0.29, 0.35, 0.4
    distSonar_far = fuzz.trapmf(x_distSonar, [0.39, 0.45, 0.5,0.5])#0.39, 0.45, 0.5,0.5
    #outputs
    '''
    angular_tooLeft = fuzz.trimf(x_angular, [-1.5, -1.3, -1.1])
    angular_left = fuzz.trimf(x_angular, [-1.2, -1.0, 0.0])
    angular_straight = fuzz.trimf(x_angular, [-0.1, 0, 0.1])
    angular_right = fuzz.trimf(x_angular, [0.0, 0.6, 1.4])
    angular_tooRight = fuzz.trimf(x_angular, [0.7, 1.2, 1.6])
    '''
    angular_ttLeft = fuzz.trapmf(x_angular, [-1.5,-1.5, -0.5, -0.45])
    angular_tooLeft = fuzz.trimf(x_angular, [-0.55, -0.46, -0.3])
    angular_left = fuzz.trimf(x_angular, [-0.462, -0.32, 0.0])
    angular_straight = fuzz.trimf(x_angular, [-0.004,0.0,0.005])
    angular_right = fuzz.trimf(x_angular, [0.0, 0.34, 0.47])
    angular_tooRight = fuzz.trimf(x_angular, [0.35, 0.475, 0.52])
    angular_ttRight = fuzz.trapmf(x_angular, [0.48, 0.53, 1.5, 1.5])

    #linear_tooSlow = fuzz.trimf(x_linear, [0, 0, 5])
    linear_stop = fuzz.trimf(x_linear, [-0.01, 0, 0.01])
    linear_slow = fuzz.trimf(x_linear, [0.03, 0.09, 0.15])#0, 0.4, 0.8
    linear_normal = fuzz.trimf(x_linear, [0.1, 0.2, 0.3])#0.5, 1.2, 1.5
    #linear_fast = fuzz.trimf(x_linear, [0.2, 0.3, 0.5])#1.3, 1.7, 2
    linear_back = fuzz.trimf(x_linear, [-0.05, -0.05, 0])#-0.3, -0.1, 0


    # Visualize these universes and membership functions
    '''
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

    ax0.plot(x_distSonar, distSonar_tooClose, 'b', linewidth=1.5, label='tooClose')
    ax0.plot(x_distSonar, distSonar_close, 'g', linewidth=1.5, label='Close')
    ax0.plot(x_distSonar, distSonar_medium, 'r', linewidth=1.5, label='Medium')
    ax0.plot(x_distSonar, distSonar_far, 'y', linewidth=1.5, label='Far')
    ax0.set_title('Sonar distance')
    ax0.legend()

    ax1.plot(x_angular, angular_ttLeft, 'c', linewidth=1.5, label='ttLeft')
    ax1.plot(x_angular, angular_tooLeft, 'b', linewidth=1.5, label='tooLeft')
    ax1.plot(x_angular, angular_left, 'g', linewidth=1.5, label='Left')
    ax1.plot(x_angular, angular_straight, 'r', linewidth=1.5, label='Straight')
    ax1.plot(x_angular, angular_right, 'y', linewidth=1.5, label='Right')
    ax1.plot(x_angular, angular_tooRight, 'k', linewidth=1.5, label='tooRight')
    ax1.plot(x_angular, angular_ttRight, 'm', linewidth=1.5, label='ttRight')
    ax1.set_title('Angular Velocity')
    ax1.legend()

    ax2.plot(x_linear, linear_stop, 'b', linewidth=1.5, label='Stop')
    ax2.plot(x_linear, linear_slow, 'g', linewidth=1.5, label='Slow')
    ax2.plot(x_linear, linear_normal, 'r', linewidth=1.5, label='Normal')
    #ax2.plot(x_linear, linear_fast, 'y', linewidth=1.5, label='Fast')
    ax2.plot(x_linear, linear_back, 'k', linewidth=1.5, label='Back')
    ax2.set_title('Linear Velocity')
    ax2.legend()

    # Turn off top/right axes
    for ax in (ax0, ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout()
    #plt.show()
    '''
    #########################################################################################

    # We need the activation of our fuzzy membership functions at these values.

    #Sonar 1 - mais à esquerda
    s1_distance_tooClose = fuzz.interp_membership(x_distSonar, distSonar_tooClose, s1Distance)
    s1_distance_close = fuzz.interp_membership(x_distSonar, distSonar_close, s1Distance)
    s1_distance_medium = fuzz.interp_membership(x_distSonar, distSonar_medium, s1Distance)
    s1_distance_far = fuzz.interp_membership(x_distSonar, distSonar_far, s1Distance)

    #Sonar 2 - rontal + à esquerda
    s2_distance_tooClose = fuzz.interp_membership(x_distSonar, distSonar_tooClose, s2Distance)
    s2_distance_close = fuzz.interp_membership(x_distSonar, distSonar_close, s2Distance)
    s2_distance_medium = fuzz.interp_membership(x_distSonar, distSonar_medium, s2Distance)
    s2_distance_far = fuzz.interp_membership(x_distSonar, distSonar_far, s2Distance)

    #Sonar 3 - frontal + à direita
    s3_distance_tooClose = fuzz.interp_membership(x_distSonar, distSonar_tooClose, s3Distance)
    s3_distance_close = fuzz.interp_membership(x_distSonar, distSonar_close, s3Distance)
    s3_distance_medium = fuzz.interp_membership(x_distSonar, distSonar_medium, s3Distance)
    s3_distance_far = fuzz.interp_membership(x_distSonar, distSonar_far, s3Distance)

    #Sonar 4 - mais à direita
    s4_distance_tooClose = fuzz.interp_membership(x_distSonar, distSonar_tooClose, s4Distance)
    s4_distance_close = fuzz.interp_membership(x_distSonar, distSonar_close, s4Distance)
    s4_distance_medium = fuzz.interp_membership(x_distSonar, distSonar_medium, s4Distance)
    s4_distance_far = fuzz.interp_membership(x_distSonar, distSonar_far, s4Distance)

    # Now we take our rules and apply them.

    # Now we apply this by clipping the top off the corresponding output
    # membership function with `np.fmin`

    #Sonar 1 - mapeamento para todas as distancias de entrada
    active_rule1 = np.fmin(s1_distance_far, linear_normal)
    active_rule2 = np.fmin(s1_distance_far, angular_straight)

    active_rule3 = np.fmin(s1_distance_medium, linear_slow)
    active_rule4 = np.fmin(s1_distance_medium, angular_right)

    active_rule5 = np.fmin(s1_distance_close, linear_slow)
    active_rule6 = np.fmin(s1_distance_close, angular_tooRight)

    active_rule7 = np.fmin(s1_distance_tooClose, linear_back)
    active_rule8 = np.fmin(s1_distance_tooClose, angular_right)

    #Sonar 2 - mapeamento para todas as distancias de entrada
    active_rule9 = np.fmin(s2_distance_far, linear_normal)
    active_rule10 = np.fmin(s2_distance_far, angular_straight)

    active_rule11 = np.fmin(s2_distance_medium, linear_slow)
    active_rule12 = np.fmin(s2_distance_medium, angular_right)

    active_rule13 = np.fmin(s2_distance_close, linear_stop)
    active_rule14 = np.fmin(s2_distance_close, angular_right)

    active_rule15 = np.fmin(s2_distance_tooClose, linear_back)
    active_rule16 = np.fmin(s2_distance_tooClose, angular_tooRight)

    #Sonar 3 - mapeamento para todas as distancias de entrada
    active_rule17 = np.fmin(s3_distance_far, linear_normal)
    active_rule18 = np.fmin(s3_distance_far, angular_straight)

    active_rule19 = np.fmin(s3_distance_medium, linear_slow)
    active_rule20 = np.fmin(s3_distance_medium, angular_left)

    active_rule21 = np.fmin(s3_distance_close, linear_stop)
    active_rule22 = np.fmin(s3_distance_close, angular_left)

    active_rule23 = np.fmin(s3_distance_tooClose, linear_back)
    active_rule24 = np.fmin(s3_distance_tooClose, angular_tooLeft)

    #Sonar 4 - mapeamento para todas as distancias de entrada
    active_rule25 = np.fmin(s4_distance_far, linear_normal)
    active_rule26 = np.fmin(s4_distance_far, angular_straight)

    active_rule27 = np.fmin(s4_distance_medium, linear_slow)
    active_rule28 = np.fmin(s4_distance_medium, angular_left)

    active_rule29 = np.fmin(s4_distance_close, linear_slow)
    active_rule30 = np.fmin(s4_distance_close, angular_tooLeft)

    active_rule31 = np.fmin(s4_distance_tooClose, linear_back)
    active_rule32 = np.fmin(s4_distance_tooClose, angular_left)

    #novas - quinas
    active_rule33 = np.fmin(linear_back, np.fmin(np.fmin(s1_distance_tooClose, s4_distance_tooClose), np.fmin(np.fmax(s2_distance_close, s2_distance_tooClose), np.fmax(s3_distance_close, s3_distance_tooClose))))
    active_rule34 = np.fmin(angular_ttRight, np.fmin(np.fmin(s1_distance_tooClose, s4_distance_tooClose), np.fmin(np.fmax(s2_distance_close, s2_distance_tooClose), np.fmax(s3_distance_close, s3_distance_tooClose))))

    #active_rule37 = np.fmin(linear_back, np.fmin(np.fmin(s1_distance_tooClose, s2_distance_tooClose),np.fmin(s3_distance_tooClose, s4_distance_tooClose)))
    #active_rule38 = np.fmin(angular_ttRight, np.fmin(np.fmin(s1_distance_tooClose, s2_distance_tooClose),np.fmin(s3_distance_tooClose, s4_distance_tooClose)))

    active_rule35 = np.fmin(np.fmin(s2_distance_tooClose, s3_distance_tooClose),linear_back)
    #active_rule36 = np.fmin(np.fmin(s2_distance_tooClose, s3_distance_tooClose), angular_ttRight)
    active_rule36 = np.fmin(np.fmin(np.fmin(s2_distance_tooClose, s3_distance_tooClose), np.fmin(s4_distance_far, np.fmax(s1_distance_medium, s1_distance_far))), angular_ttRight) #np.fmax(s4_distance_far, s1_distance_medium)
    active_rule37 = np.fmin(np.fmin(np.fmin(s2_distance_tooClose, s3_distance_tooClose), np.fmax(s1_distance_far, np.fmax(s4_distance_medium, s4_distance_far))), angular_ttLeft) #np.fmax(s1_distance_far, s4_distance_medium)

    #objetos na diagonal
    active_rule38 = np.fmin(np.fmin(s1_distance_far, np.fmax(s4_distance_tooClose, s4_distance_close)), linear_back)
    active_rule39 = np.fmin(np.fmin(s4_distance_tooClose, np.fmin(np.fmax(s2_distance_tooClose, s2_distance_close), np.fmax(s3_distance_tooClose, s3_distance_close))), angular_ttLeft)
    active_rule40 = np.fmin(np.fmin(s1_distance_tooClose, np.fmin(np.fmax(s2_distance_tooClose, s2_distance_close), np.fmax(s3_distance_tooClose, s3_distance_close))), angular_ttRight)

    active_rule41 = np.fmin(linear_slow,
                      np.fmin(
                                  np.fmax(s1_distance_tooClose, s1_distance_close),
                                  np.fmax(s4_distance_tooClose, s4_distance_close)),
                              )
    active_rule42 = np.fmin(angular_straight,
                      np.fmin(
                                  np.fmax(s1_distance_tooClose, s1_distance_close),
                                  np.fmax(s4_distance_tooClose, s4_distance_close)),
                              )
    active_rule43 = np.fmin(np.fmin(np.fmin(np.fmax(s1_distance_close, s1_distance_tooClose), np.fmax(s2_distance_close, s2_distance_tooClose)), np.fmin(np.fmax(s3_distance_close, s3_distance_tooClose), np.fmax(s4_distance_close, s4_distance_tooClose))), linear_back)
    active_rule44 = np.fmin(np.fmin(np.fmin(np.fmax(s1_distance_close, s1_distance_tooClose), np.fmax(s2_distance_close, s2_distance_tooClose)), np.fmin(np.fmax(s3_distance_close, s3_distance_tooClose), np.fmax(s4_distance_close, s4_distance_tooClose))), angular_straight)
    ######################### Depois de todas as regras ########################################

    linear0 = np.zeros_like(x_linear)
    angular0 = np.zeros_like(x_angular)
    '''
    # Visualize this - Linear Speed
    fig, (ax0, ax1,) = plt.subplots(nrows=2, figsize=(8, 9))

    ax0.fill_between(x_linear, linear0, active_rule7, facecolor='b', alpha=0.7)
    ax0.fill_between(x_linear, linear0, active_rule15, facecolor='b', alpha=0.7)
    ax0.fill_between(x_linear, linear0, active_rule23, facecolor='b', alpha=0.7)
    ax0.fill_between(x_linear, linear0, active_rule31, facecolor='b', alpha=0.7)
    ax0.fill_between(x_linear, linear0, active_rule33, facecolor='b', alpha=0.7)
    ax0.fill_between(x_linear, linear0, active_rule35, facecolor='b', alpha=0.7)
    ax0.plot(x_linear, linear_back, 'b', linewidth=0.5, linestyle='--', )

    ax0.fill_between(x_linear, linear0, active_rule3, facecolor='r', alpha=0.7)
    ax0.fill_between(x_linear, linear0, active_rule5, facecolor='r', alpha=0.7)
    ax0.fill_between(x_linear, linear0, active_rule11, facecolor='r', alpha=0.7)
    ax0.fill_between(x_linear, linear0, active_rule19, facecolor='r', alpha=0.7)
    ax0.fill_between(x_linear, linear0, active_rule27, facecolor='r', alpha=0.7)
    ax0.fill_between(x_linear, linear0, active_rule29, facecolor='r', alpha=0.7)
    ax0.plot(x_linear, linear_slow, 'r', linewidth=0.5, linestyle='--')

    ax0.fill_between(x_linear, linear0, active_rule1, facecolor='g', alpha=0.7)
    ax0.fill_between(x_linear, linear0, active_rule9, facecolor='g', alpha=0.7)
    ax0.fill_between(x_linear, linear0, active_rule17, facecolor='g', alpha=0.7)
    ax0.fill_between(x_linear, linear0, active_rule25, facecolor='g', alpha=0.7)
    ax0.plot(x_linear, linear_normal, 'g', linewidth=0.5, linestyle='--')

    ax0.fill_between(x_linear, linear0, active_rule13, facecolor='y', alpha=0.7)
    ax0.fill_between(x_linear, linear0, active_rule21, facecolor='y', alpha=0.7)
    ax0.plot(x_linear, linear_stop, 'y', linewidth=0.5, linestyle='--')

    ax0.set_title('Output membership activity for Linear Speed')


    # Visualize this - Angular Speed
    ax1.fill_between(x_angular, angular0, active_rule24, facecolor='b', alpha=0.7)
    ax1.fill_between(x_angular, angular0, active_rule30, facecolor='b', alpha=0.7)
    ax1.plot(x_angular, angular_tooLeft, 'b', linewidth=0.5, linestyle='--', )

    ax1.fill_between(x_angular, angular0, active_rule20, facecolor='g', alpha=0.7)
    ax1.fill_between(x_angular, angular0, active_rule22, facecolor='g', alpha=0.7)
    ax1.fill_between(x_angular, angular0, active_rule28, facecolor='g', alpha=0.7)
    ax1.fill_between(x_angular, angular0, active_rule32, facecolor='g', alpha=0.7)
    ax1.plot(x_angular, angular_left, 'g', linewidth=0.5, linestyle='--')

    ax1.fill_between(x_angular, angular0, active_rule2, facecolor='r', alpha=0.7)
    ax1.fill_between(x_angular, angular0, active_rule10, facecolor='r', alpha=0.7)
    ax1.fill_between(x_angular, angular0, active_rule18, facecolor='r', alpha=0.7)
    ax1.fill_between(x_angular, angular0, active_rule26, facecolor='r', alpha=0.7)
    ax1.plot(x_angular, angular_straight, 'r', linewidth=0.5, linestyle='--')

    ax1.fill_between(x_angular, angular0, active_rule4, facecolor='k', alpha=0.7)
    ax1.fill_between(x_angular, angular0, active_rule8, facecolor='k', alpha=0.7)
    ax1.fill_between(x_angular, angular0, active_rule12, facecolor='k', alpha=0.7)
    ax1.fill_between(x_angular, angular0, active_rule14, facecolor='k', alpha=0.7)
    ax1.plot(x_angular, angular_right, 'k', linewidth=0.5, linestyle='--')

    ax1.fill_between(x_angular, angular0, active_rule6, facecolor='y', alpha=0.7)
    ax1.fill_between(x_angular, angular0, active_rule16, facecolor='y', alpha=0.7)
    ax1.plot(x_angular, angular_tooRight, 'y', linewidth=0.5, linestyle='--')

    ax1.fill_between(x_angular, angular0, active_rule34, facecolor='m', alpha=0.7)
    ax1.fill_between(x_angular, angular0, active_rule36, facecolor='m', alpha=0.7)
    ax1.plot(x_angular, angular_ttRight, 'm', linewidth=0.5, linestyle='--')

    ax1.fill_between(x_angular, angular0, active_rule37, facecolor='c', alpha=0.7)
    ax1.plot(x_angular, angular_ttLeft, 'm', linewidth=0.5, linestyle='--')

    ax1.set_title('Output membership activity for Angular Speed')


    # Turn off top/right axes
    for ax in (ax0, ax1,):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout()
    #plt.show()
    '''
    ########################## Rule aggregation and Defuzzification ###################################

    #Linear Speed
    # Aggregate all output membership functions together
    #aggregatedLinear = np.fmax(np.fmax(np.fmax(np.fmax(active_rule1, active_rule3), np.fmax(active_rule5, active_rule7)), np.fmax(np.fmax(active_rule9, active_rule11), np.fmax(active_rule13, active_rule15))), np.fmax(np.fmax(np.fmax(active_rule17, active_rule19), np.fmax(active_rule21, active_rule23)), np.fmax(np.fmax(active_rule25, active_rule27), np.fmax(active_rule29, active_rule31))))
    #aggregatedLinear = np.fmax(active_rule33, np.fmax(np.fmax(np.fmax(np.fmax(active_rule1, active_rule3), np.fmax(active_rule5, active_rule7)),
                    #np.fmax(np.fmax(active_rule9, active_rule11), np.fmax(active_rule13, active_rule15))), np.fmax(np.fmax(np.fmax(active_rule17,
                        #active_rule19), np.fmax(active_rule21, active_rule23)), np.fmax(np.fmax(active_rule25, active_rule27), np.fmax(active_rule29, active_rule31)))))
    aggregatedLinear = np.fmax(active_rule43, np.fmax(active_rule41, np.fmax(active_rule38, np.fmax(active_rule35,
              np.fmax(active_rule33,
                np.fmax(
                  np.fmax(
                    np.fmax(
                      np.fmax(active_rule1, active_rule3),
                      np.fmax(active_rule5, active_rule7)),
                    np.fmax(
                      np.fmax(active_rule9, active_rule11),
                      np.fmax(active_rule13, active_rule15))),
                  np.fmax(
                    np.fmax(
                      np.fmax(active_rule17, active_rule19),
                      np.fmax(active_rule21, active_rule23)),
                    np.fmax(
                      np.fmax(active_rule25, active_rule27),
                      np.fmax(active_rule29, active_rule31)))))))))
    # Calculate defuzzified result
    linear = fuzz.defuzz(x_linear, aggregatedLinear, 'centroid')
    linear_activation = fuzz.interp_membership(x_linear, aggregatedLinear, linear)  # for plot

    #Angular Speed
    # Aggregate all output membership functions together
    #aggregatedAngular = np.fmax(np.fmax(np.fmax(np.fmax(active_rule2, active_rule4), np.fmax(active_rule6, active_rule8)), np.fmax(np.fmax(active_rule10, active_rule12), np.fmax(active_rule14, active_rule16))), np.fmax(np.fmax(np.fmax(active_rule18, active_rule20), np.fmax(active_rule22, active_rule24)), np.fmax(np.fmax(active_rule26, active_rule28), np.fmax(active_rule30, active_rule32))))
    #aggregatedAngular = np.fmax(active_rule34, np.fmax(np.fmax(np.fmax(np.fmax(active_rule2, active_rule4), np.fmax(active_rule6, active_rule8)), np.fmax(np.fmax(active_rule10, active_rule12), np.fmax(active_rule14, active_rule16))), np.fmax(np.fmax(np.fmax(active_rule18, active_rule20), np.fmax(active_rule22, active_rule24)), np.fmax(np.fmax(active_rule26, active_rule28), np.fmax(active_rule30, active_rule32)))))
    aggregatedAngular = np.fmax(active_rule44, np.fmax(active_rule42, np.fmax(active_rule40, np.fmax(active_rule39, np.fmax(active_rule37,
        np.fmax(active_rule36,
              np.fmax(active_rule34,
                np.fmax(
                  np.fmax(
                    np.fmax(
                      np.fmax(active_rule2, active_rule4),
                      np.fmax(active_rule6, active_rule8)),
                    np.fmax(
                      np.fmax(active_rule10, active_rule12),
                      np.fmax(active_rule14, active_rule16))),
                  np.fmax(
                    np.fmax(
                      np.fmax(active_rule18, active_rule20),
                      np.fmax(active_rule22, active_rule24)),
                    np.fmax(
                      np.fmax(active_rule26, active_rule28),
                      np.fmax(active_rule30, active_rule32)))))))))))
    # Calculate defuzzified result
    angular = fuzz.defuzz(x_angular, aggregatedAngular, 'centroid')
    angular_activation = fuzz.interp_membership(x_angular, aggregatedAngular, angular)  # for plot

    '''
    # Visualize this
    fig, (ax0, ax1,) = plt.subplots(nrows=2, figsize=(8, 9))

    ax0.plot(x_linear, linear_stop, 'b', linewidth=0.5, linestyle='--', )
    ax0.plot(x_linear, linear_slow, 'g', linewidth=0.5, linestyle='--')
    ax0.plot(x_linear, linear_normal, 'r', linewidth=0.5, linestyle='--')
    #ax0.plot(x_linear, linear_fast, 'm', linewidth=0.5, linestyle='--')
    ax0.plot(x_linear, linear_back, 'y', linewidth=0.5, linestyle='--')
    ax0.fill_between(x_linear, linear0, aggregatedLinear, facecolor='Orange', alpha=0.7)
    ax0.plot([linear, linear], [0, linear_activation], 'k', linewidth=1.5, alpha=0.9)
    ax0.set_title('Aggregated membership for Linear Speed and result (line)')

    ax1.plot(x_angular, angular_ttLeft, 'c', linewidth=0.5, linestyle='--', )
    ax1.plot(x_angular, angular_tooLeft, 'b', linewidth=0.5, linestyle='--', )
    ax1.plot(x_angular, angular_left, 'g', linewidth=0.5, linestyle='--')
    ax1.plot(x_angular, angular_straight, 'r', linewidth=0.5, linestyle='--')
    ax1.plot(x_angular, angular_right, 'm', linewidth=0.5, linestyle='--')
    ax1.plot(x_angular, angular_tooRight, 'y', linewidth=0.5, linestyle='--')
    ax1.plot(x_angular, angular_ttRight, 'c', linewidth=0.5, linestyle='--', )
    ax1.fill_between(x_angular, angular0, aggregatedAngular, facecolor='Orange', alpha=0.7)
    ax1.plot([angular, angular], [0, angular_activation], 'k', linewidth=1.5, alpha=0.9)
    ax1.set_title('Aggregated membership for Angular Speed and result (line)')

    # Turn off top/right axes
    for ax in (ax0, ax1,):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout()
    #plt.show()
    '''
    return (linear, angular)
