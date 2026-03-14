#  Copyright (c) 2021 European Union
#  Licensed under the EUPL, Version 1.2

import numpy as np
from utility import fuzzy_functs as ff

# ============================================================
# Available models
# ============================================================
models_list = ['FSM', 'CC_human_driver', 'RSS', 'Reg157', 'HMM']

# ============================================================
# Generic utility functions
# ============================================================

def model_react(ego_veh, speed_log, freq):
    return max(speed_log - ego_veh.max_a / freq, 0), 0


# ============================================================
# Distance calculation functions for each model
# ============================================================

def FSM_distance(imaginary_veh, ur):
    """Calculate initial distance for FSM model"""
    return ur * 2 + 10  # Simplified formula for demo

def CC_distance(imaginary_veh, ur):
    """Calculate initial distance for CC model"""
    return ur * 1.5 + 8

def RSS_distance(imaginary_veh, ur):
    """Calculate initial distance for RSS model"""
    return ur * 2.5 + 12


# ============================================================
# CC Human Driver Model
# ============================================================

def CC_calc_dist_to_react(lane_width, cut_in_width, ego_dist_to_lane):
    lane_marking_to_center_cut_in = (lane_width - cut_in_width) / 2
    wp_dist_to_react = lane_marking_to_center_cut_in - 0.72 - 0.375
    return ego_dist_to_lane + wp_dist_to_react


def CC_check_safety(ego_veh, cutting_in_veh, speed_log, speed_lat, freq, i):
    if ego_veh.pos_profile_long[i] > cutting_in_veh.pos_profile_long[i]:
        return True

    lat_gap = abs(ego_veh.pos_profile_lat[i] - cutting_in_veh.pos_profile_lat[i]) \
              - ego_veh.width / 2 - cutting_in_veh.width / 2
    if lat_gap > 0:
        return True

    lon_gap = abs(ego_veh.pos_profile_long[i] - cutting_in_veh.pos_profile_long[i]) \
              - ego_veh.length / 2 - cutting_in_veh.length / 2

    ttc = lon_gap / (ego_veh.speed_profile_long[i] - cutting_in_veh.speed_profile_long[i])
    if abs(ttc) > ego_veh.CC_critical_ttc:
        return True

    return False

def CC_check_safety_cut_in(ego_veh, cutting_in_veh, speed_log, speed_lat, freq, i):
    return CC_check_safety(ego_veh, cutting_in_veh, speed_log, speed_lat, freq, i)


def CC_react(ego_veh, speed_log, freq):
    if ego_veh.CC_rt_counter > 0:
        ego_veh.CC_rt_counter -= 1 / freq
        ego_veh.deceleration = ego_veh.CC_release_deceleration
    else:
        ego_veh.deceleration = min(
            ego_veh.deceleration + ego_veh.CC_min_jerk / freq,
            ego_veh.CC_max_deceleration
        )
    return max(speed_log - ego_veh.deceleration / freq, 0), 0


# ============================================================
# RSS Model
# ============================================================

def RSS_check_safety(ego_veh, cutting_in_veh, speed_log, speed_lat, freq, i):
    if ego_veh.pos_profile_long[i] > cutting_in_veh.pos_profile_long[i]:
        return True

    d_safe = speed_log * ego_veh.RSS_rt \
             + ego_veh.max_a * ego_veh.RSS_rt ** 2 / 2 \
             + (speed_log + ego_veh.max_a * ego_veh.RSS_rt) ** 2 / (2 * ego_veh.max_d) \
             - cutting_in_veh.speed_profile_long[i] ** 2 / (2 * cutting_in_veh.max_d)

    gap = abs(ego_veh.pos_profile_long[i] - cutting_in_veh.pos_profile_long[i]) \
          - ego_veh.length / 2 - cutting_in_veh.length / 2

    if gap < d_safe:
        return False
    return True


def RSS_react(ego_veh, speed_log, freq):
    if ego_veh.RSS_rt_counter > 0:
        ego_veh.RSS_rt_counter -= 1 / freq
        return speed_log, 0

    ego_veh.deceleration = min(
        ego_veh.deceleration + ego_veh.RSS_min_jerk / freq,
        ego_veh.RSS_max_deceleration
    )
    return max(speed_log - ego_veh.deceleration / freq, 0), 0


# ============================================================
# Reg 157 Model
# ============================================================

def Reg157_check_safety(ego_veh, cutting_in_veh, speed_log, speed_lat, freq, i):
    return Reg157_check_safety_cut_in(ego_veh, cutting_in_veh, speed_log, speed_lat, freq, i)

def Reg157_check_safety_cut_in(ego_veh, cutting_in_veh, speed_log, speed_lat, freq, i):
    if ego_veh.pos_profile_long[i] > cutting_in_veh.pos_profile_long[i]:
        return True

    lat_gap = abs(ego_veh.pos_profile_lat[i] - cutting_in_veh.pos_profile_lat[i]) \
              - ego_veh.width / 2 - cutting_in_veh.width / 2

    if lat_gap > ego_veh.Reg157_lat_safe_dist:
        return True

    lon_gap = abs(ego_veh.pos_profile_long[i] - cutting_in_veh.pos_profile_long[i]) \
              - ego_veh.length / 2 - cutting_in_veh.length / 2

    ttc = lon_gap / (ego_veh.speed_profile_long[i] - cutting_in_veh.speed_profile_long[i])

    if ttc > (ego_veh.speed_profile_long[i] / (2 * ego_veh.Reg157_max_deceleration)
              + ego_veh.Reg157_rt):
        return True

    return False


def Reg157_react(ego_veh, speed_log, freq):
    if ego_veh.Reg157_rt_counter > 0:
        ego_veh.Reg157_rt_counter -= 1 / freq
        return speed_log, 0
    return max(speed_log - ego_veh.Reg157_max_deceleration / freq, 0), 0


# ============================================================
# FSM (Fuzzy Safety Model)
# ============================================================

def FSM_check_safety(ego_veh, cutting_in_veh, speed_log, speed_lat, freq, i):
    if ego_veh.pos_profile_long[i] > cutting_in_veh.pos_profile_long[i]:
        return True

    dist = abs(ego_veh.pos_profile_long[i] - cutting_in_veh.pos_profile_long[i]) \
           - ego_veh.length / 2 - cutting_in_veh.length / 2

    ar = (ego_veh.speed_profile_long[i] - ego_veh.speed_profile_long[i - 1]) * freq

    cfs = ff.CFS(
        dist,
        ego_veh.speed_profile_long[i],
        cutting_in_veh.speed_profile_long[i],
        ego_veh.FSM_rt,
        ego_veh.FSM_br_min,
        ego_veh.FSM_br_max,
        ego_veh.FSM_bl,
        ar
    )

    pfs = ff.PFS(
        dist,
        ego_veh.speed_profile_long[i],
        cutting_in_veh.speed_profile_long[i],
        ego_veh.FSM_rt,
        ego_veh.FSM_br_min,
        ego_veh.FSM_br_max,
        ego_veh.FSM_bl,
        ego_veh.FSM_margin_dist,
        ego_veh.FSM_margin_safe_dist
    )

    ego_veh.cfs = cfs
    ego_veh.pfs = pfs

    return (cfs + pfs) == 0


def FSM_react(ego_veh, speed_log, freq):
    if ego_veh.FSM_rt_counter > 0:
        ego_veh.FSM_rt_counter -= 1 / freq
        return speed_log, 0

    acc = ego_veh.cfs * ego_veh.FSM_br_max if ego_veh.cfs > 0 else ego_veh.pfs * ego_veh.FSM_br_min

    ego_veh.deceleration = min(
        ego_veh.deceleration + ego_veh.CC_min_jerk / freq,
        acc
    )

    return max(speed_log - ego_veh.deceleration / freq, 0), 0


# ============================================================
# HMM Model (Hidden Markov Model)
# ============================================================

class SimpleHMM:
    """
    Hidden Markov Model for safety state prediction
    States:
    0 = SAFE
    1 = WARNING  
    2 = DANGER
    """
    
    def __init__(self):
        # Transition probabilities between states
        self.transition = np.array([
            [0.8, 0.2, 0.0],   # From SAFE
            [0.2, 0.6, 0.2],   # From WARNING
            [0.0, 0.3, 0.7]    # From DANGER
        ])
        self.state_prob = np.array([1.0, 0.0, 0.0])  # Start in SAFE state
        self.safe_threshold = 0.3  # Threshold for DANGER state
    
    def emission_probability(self, distance, rel_speed, ttc):
        """
        Calculate emission probabilities based on observations
        """
        # Normalize inputs
        norm_dist = min(distance / 50.0, 1.0)  # Max distance considered 50m
        norm_rel_speed = min(abs(rel_speed) / 20.0, 1.0)  # Max relative speed 20 m/s
        norm_ttc = min(ttc / 10.0, 1.0) if ttc > 0 else 0
        
        # Higher distance and TTC -> more likely to be SAFE
        # Higher relative speed -> more likely to be DANGER
        safe_score = 0.6 * norm_dist + 0.4 * norm_ttc
        danger_score = 0.7 * (1 - norm_ttc) + 0.3 * norm_rel_speed
        warning_score = 0.5 * (1 - abs(safe_score - danger_score))
        
        # Normalize to probabilities
        scores = np.array([safe_score, warning_score, danger_score])
        scores = np.clip(scores, 0.1, 0.9)  # Avoid zero probabilities
        return scores / np.sum(scores)
    
    def update(self, distance, rel_speed, ttc):
        """
        Update state probabilities based on new observation
        """
        # Get emission probabilities
        emission = self.emission_probability(distance, rel_speed, ttc)
        
        # Predict next state (transition)
        predicted = self.state_prob @ self.transition
        
        # Update with observation (Bayesian update)
        self.state_prob = emission * predicted
        self.state_prob /= np.sum(self.state_prob)  # Normalize
        
        return self.state_prob
    
    def is_safe(self):
        """Return True if DANGER probability is below threshold"""
        return self.state_prob[2] < self.safe_threshold


def HMM_distance(imaginary_veh, ur):
    """Calculate initial distance for HMM model"""
    return ur * 1.8 + 10  # Conservative distance


def HMM_check_safety(ego_veh, cutting_in_veh, speed_log, speed_lat, freq, i):
    """Check safety for car-following scenario"""
    if ego_veh.pos_profile_long[i] > cutting_in_veh.pos_profile_long[i]:
        return True
    
    # Calculate distance and relative speed
    distance = abs(ego_veh.pos_profile_long[i] - cutting_in_veh.pos_profile_long[i]) \
               - ego_veh.length / 2 - cutting_in_veh.length / 2
    
    rel_speed = ego_veh.speed_profile_long[i] - cutting_in_veh.speed_profile_long[i]
    
    # Calculate Time To Collision (TTC)
    if rel_speed > 0:
        ttc = distance / rel_speed if distance > 0 else 0
    else:
        ttc = 10  # Large TTC if moving away
    
    # Update HMM
    state_probs = ego_veh.hmm_model.update(distance, rel_speed, ttc)
    
    
    # Check if in DANGER state
    return ego_veh.hmm_model.is_safe()


def HMM_check_safety_cut_in(ego_veh, cutting_in_veh, speed_log, speed_lat, freq, i):
    """Check safety for cut-in scenario"""
    if ego_veh.pos_profile_long[i] > cutting_in_veh.pos_profile_long[i]:
        return True
    
    # Calculate both longitudinal and lateral metrics
    lon_distance = abs(ego_veh.pos_profile_long[i] - cutting_in_veh.pos_profile_long[i]) \
                   - ego_veh.length / 2 - cutting_in_veh.length / 2
    
    lat_distance = abs(ego_veh.pos_profile_lat[i] - cutting_in_veh.pos_profile_lat[i]) \
                   - ego_veh.width / 2 - cutting_in_veh.width / 2
    
    rel_speed = ego_veh.speed_profile_long[i] - cutting_in_veh.speed_profile_long[i]
    
    # Use minimum of longitudinal and lateral TTC
    ttc_lon = lon_distance / rel_speed if rel_speed > 0 and lon_distance > 0 else 10
    ttc_lat = lat_distance / abs(cutting_in_veh.speed_profile_lat[i]) if lat_distance > 0 else 10
    
    ttc = min(ttc_lon, ttc_lat)
    
    # Use combined distance (weighted sum)
    combined_distance = 0.7 * lon_distance + 0.3 * lat_distance
    
    # Update HMM
    state_probs = ego_veh.hmm_model.update(combined_distance, rel_speed, ttc)
    
    # Check if in DANGER state
    return ego_veh.hmm_model.is_safe()


def HMM_react(ego_veh, speed_log, freq):
    """Reaction function for HMM model"""
    if ego_veh.HMM_rt_counter > 0:
        ego_veh.HMM_rt_counter -= 1 / freq
        return speed_log, 0
    
    # Get current danger probability
    danger_prob = ego_veh.hmm_model.state_prob[2]
    
    # Calculate deceleration based on danger probability
    # More danger -> stronger braking
    base_deceleration = ego_veh.HMM_max_deceleration * danger_prob
    
    # Add some smoothing
    ego_veh.deceleration = min(
        ego_veh.deceleration + ego_veh.HMM_min_jerk / freq,
        base_deceleration
    )
    
    return max(speed_log - ego_veh.deceleration / freq, 0), 0,