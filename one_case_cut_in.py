#  Copyright (c) 2021 European Union
#  *
#  Licensed under the EUPL, Version 1.2 or – as soon they will be approved by the
#  European Commission – subsequent versions of the EUPL (the "Licence");
#  You may not use this work except in compliance with the Licence.
#  You may obtain a copy of the Licence at:
#  *
#  https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12
#  *
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the Licence is distributed on an "AS IS" basis,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the Licence for the specific language governing permissions and limitations
#  under the Licence.
#
#

import numpy as np
import matplotlib.pyplot as plt
from post_processing import live_graph as li
from . import movement as mvt
from . import models as md
from . import global_parameters as gp
from .models import SimpleHMM  # Add import


def run_one_case(use_model, **kwargs):

    length = gp.length
    width = gp.width
    iterations = gp.iterations
    freq = gp.freq

    init_long_speed_ego = float(kwargs.get('initial_speed', 50))
    init_long_speed_c = float(kwargs.get('obstacle_speed', 20))
    lateral_speed = float(kwargs.get('lateral_speed', -1.))
    long_dist = float(kwargs.get('front_distance', 50))

    print('Starting speed cut in: ' + str(init_long_speed_ego) + ' (km/h)')
    print('Obstacle speed cut in: ' + str(init_long_speed_c) + ' (km/h)')
    print('Lateral speed cut in: ' + str(lateral_speed) + ' (m/s)')
    print('Front distance cut in: ' + str(long_dist) + ' (m)')

    init_long_speed_c = init_long_speed_c/3.6
    init_long_speed_ego = init_long_speed_ego/3.6

    init_pos_c = np.array([long_dist + length, 1.6 + width])
    init_pos_ego = np.array([0, 0])

    if use_model == 'FSM':
        check = md.FSM_check_safety
        react = md.FSM_react
        print("Selected model: FSM")
    elif use_model == 'CC_human_driver':
        check = md.CC_check_safety_cut_in
        react = md.CC_react
        print("Selected model: CC_human_driver")
    elif use_model == 'RSS':
        check = md.RSS_check_safety
        react = md.RSS_react
        print("Selected model: RSS")
    elif use_model == 'Reg157':
        check = md.Reg157_check_safety_cut_in
        react = md.Reg157_react
        print("Selected model: Reg157")
    elif use_model == 'HMM':
        check = md.HMM_check_safety_cut_in
        react = md.HMM_react
        print("Selected model: HMM")
    else:
        print("Error, unsupported safety model\n")
        exit(4)

    cut_in_veh = mvt.create_profile_cutting_in(init_pos_c, init_long_speed_c, lateral_speed, iterations, freq)
    ego_veh = mvt.create_profile_cutting_in(init_pos_ego, init_long_speed_ego, 0, iterations, freq)

    # Initialize HMM if needed
    if use_model == 'HMM':
        ego_veh.hmm_model = SimpleHMM()

    cut_in_veh.width = width
    cut_in_veh.length = length
    ego_veh.width = width
    ego_veh.length = length

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    plt.axis('off')
    vehs = [ego_veh, cut_in_veh]

    # For HMM, store state probabilities
    hmm_states = [] if use_model == 'HMM' else None

    for i in range(iterations - 1):

        if i == 1:
            plt.pause(2)

        mvt.control(ego_veh, cut_in_veh, freq, check, react, i)
        
        # Store HMM state if applicable
        if use_model == 'HMM' and hasattr(ego_veh, 'hmm_model'):
            hmm_states.append(ego_veh.hmm_model.state_prob.copy())
            
        li.plot_map(vehs, ax1, i)
        if ego_veh.crash == 1:
            fig.patch.set_facecolor((1, 0, 0, 0.2))

    print('Crash = ', ego_veh.crash)

    ''' post processing '''
    plt.figure('Trajectory')
    plt.plot(cut_in_veh.pos_profile_long, cut_in_veh.pos_profile_lat, 'rx', label='Cut-in vehicle')
    plt.plot(ego_veh.pos_profile_long, ego_veh.pos_profile_lat, 'bx', label='Ego vehicle')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()

    plt.figure('Speed')
    plt.plot(np.linspace(0, (iterations-1)/freq, iterations), ego_veh.speed_profile_long, '-bx', label='Ego vehicle')
    plt.plot(np.linspace(0, (iterations-1)/freq, iterations), cut_in_veh.speed_profile_long, 'r', label='Cut-in vehicle')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.legend()

    plt.figure('Acceleration')
    plt.plot(np.linspace(0, (iterations-2)/freq, iterations-1), np.diff(ego_veh.speed_profile_long) * freq, '-bx', label='Ego vehicle')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s$^2$)')
    plt.legend()

    # Plot HMM state probabilities if available
    if use_model == 'HMM' and hmm_states:
        hmm_states = np.array(hmm_states)
        plt.figure('HMM State Probabilities')
        plt.plot(np.linspace(0, len(hmm_states)/freq, len(hmm_states)), hmm_states[:, 0], 'g-', label='SAFE', linewidth=2)
        plt.plot(np.linspace(0, len(hmm_states)/freq, len(hmm_states)), hmm_states[:, 1], 'y-', label='WARNING', linewidth=2)
        plt.plot(np.linspace(0, len(hmm_states)/freq, len(hmm_states)), hmm_states[:, 2], 'r-', label='DANGER', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Probability')
        plt.title('HMM State Probabilities (Cut-in)')
        plt.legend()
        plt.grid(True)

    plt.show()