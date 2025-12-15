# This file contains utility functions for replay buffer management
from gymnasium import spaces
import torch
from Buffers.types import Batch

box_obs = ['attachment_time_left', 'aux_ticks', 'center_path', 'center_path_distance', 'distance_down_track', 'energy', 'front', 'max_steer_angle', 'shield_time', 'skeed_factor', 'velocity']

seq_obs = ['items_position', 'items_type', 'karts_position', 'paths_distance', 'paths_end', 'paths_start', 'paths_width']

discrete_action_keys = ['brake', 'drift', 'fire', 'nitro', 'rescue']
continuous_action_keys =  ['acceleration', 'steer']

def pad_seq(data, Lmax):
    '''
     pad a sequence of tensors to length Lmax for each time step

     data: list of tensors shape (Seq_len, D), with size equals to the episode length
     returns:
        padded: tensor of shape (episode_length, Lmax, D)
        pad_seq_mask: tensor of shape (episode_length, Lmax) with 1 for valid items in Sequence-Type obs and 0 for padded items
    '''
    D = data[0].shape[1:]

    episode_length = len(data)
    device = data[0].device

    padded = torch.zeros((episode_length, Lmax, *D), device=device)
    pad_seq_mask = torch.zeros((episode_length, Lmax), dtype=torch.bool, device=device)

    for i, item in enumerate(data):
        L_use = min(item.shape[0], Lmax)
        if len(D) == 0:
            padded[i, :L_use] = item[:L_use]
        else:
            padded[i, :L_use, ...] = item[:L_use, ...]
        pad_seq_mask[i, :L_use] = True

    return padded, pad_seq_mask

def pad_temporal(data, pad_seq_mask):

    '''
        pad a batch of sequences to a fixed length of 1500 time steps
        
        Inputs:
            data: list of tensors shape (episode_length, Lmax, D)
                where episode_length is the length of each episode,
            pad_seq_mask: list of tensors shape (episode_length, Lmax)
        returns:
            padded: tensor of shape (1500, Lmax, D)
            mask: tensor of shape (1500, Lmax) with 1 for valid steps and 0 for padded steps
    '''
    epl, lmax = data[0].shape[0], data[0].shape[1]
    D = data[0].shape[2:]

    padded = torch.zeros((1500, lmax, *D), device=data[0].device)
    mask = torch.zeros((1500, lmax), dtype=torch.bool, device=data[0].device)

    for i, item in enumerate(data):
        epl_use = min(item.shape[0], 1500)
        padded[:epl_use, ...] = item[:epl_use, ...]
        mask[:epl_use, :] = True
        mask[:epl_use, :] = pad_seq_mask[i][:epl_use, :]

    return padded, mask

def pad_temporal_others(data):
    '''
        pad a sequence to a fixed length of 1500 time steps
        specifically for other data types like actions, rewards, dones, etc.

        inputs:
            data: list of tensors shape (episode_length, D)
    '''

    epl = len(data)
    D = data[0].shape
    padded = torch.zeros((1500, *D), device=data[0].device)
    mask = torch.zeros((1500,), dtype=torch.bool, device=data[0].device)
    epl_use = min(epl, 1500)
    padded[:epl_use, ...] = torch.stack(data[:epl_use], dim=0)
    mask[:epl_use] = True

    return padded, mask

    
def transitions_to_batch(batchSeqs, Lmax, device, box_obs=box_obs, seq_obs=seq_obs):

    keys = batchSeqs[0][0].states.keys()
    action_keys = batchSeqs[0][0].action.keys()

    # Keys being as types of observations and actions; values as Batch*TimeSteps*Dimensions

    # box-Type observations
    box_obs = {k: None for k in box_obs}
    next_box_obs = {k: None for k in box_obs}

    # Sequence-Type observations
    seq_obs = {k: None for k in seq_obs}
    seq_mask = {k: None for k in seq_obs}
    next_seq_obs = {k: None for k in seq_obs}
    next_seq_mask = {k: None for k in seq_obs}

    # actions
    disc_actions = {}
    cont_actions = {}
    cont_actions_mask = {}
    disc_actions_mask = {}

    for k in keys:
        temp_box_obs = [] 
        temp_next_box_obs = []
        temp_seqs_obs = []
        temp_seqs_mask = []
        temp_next_seq_obs = []
        temp_next_seq_mask = []
        temp_cont_actions = []
        temp_disc_actions = []

        for transitions in batchSeqs:
            # ---------- box obs ----------
            if k in box_obs:
                a = [torch.tensor(t.states[k]) for t in transitions]
                box_obs_padded, mask = pad_temporal_others(a)
                temp_box_obs.append(box_obs_padded)

                next_a = [torch.tensor(t.next_states[k]) for t in transitions]
                box_next_padded, mask_n = pad_temporal_others(next_a)
                temp_next_box_obs.append(box_next_padded)

            # ---------- seq obs ----------
            elif k in seq_obs:
                seq_list = [torch.tensor(t.states[k]).to(device) for t in transitions]
                padded_seqs, t_mask = pad_seq(seq_list, Lmax)
                temp_seqs_obs.append(padded_seqs)
                temp_seqs_mask.append(t_mask)

                next_seqs = [torch.tensor(t.next_states[k]).to(device) for t in transitions]
                padded_n, mask_n = pad_seq(next_seqs, Lmax)
                temp_next_seq_obs.append(padded_n)
                temp_next_seq_mask.append(mask_n)
        
        if k in box_obs:
            box_obs[k], next_box_obs[k] = torch.stack(temp_box_obs).to(device), torch.stack(temp_next_box_obs).to(device)

        elif k in seq_obs:        
            seq_obs[k], seq_mask[k] = pad_temporal(temp_seqs_obs, temp_seqs_mask) 
            next_seq_obs[k], next_seq_mask[k] = pad_temporal(temp_next_seq_obs, temp_next_seq_mask)

    for k in action_keys:
        
        temp_cont_actions = [] 
        temp_cont_t_mask = []
        temp_disc_actions = []
        temp_disc_t_mask = [] 
        for transitions in batchSeqs:
            if k in continuous_action_keys:
                cont_pad, cont_mask = pad_temporal_others(
                    [torch.tensor(t.action[k]).squeeze() for t in transitions]
                )
                temp_cont_actions.append(cont_pad)
                temp_cont_t_mask.append(cont_mask)


            if k in discrete_action_keys:
                disc_pad, disc_t_mask = pad_temporal_others(
                    [torch.tensor(t.action[k]).squeeze() for t in transitions]
                )
                temp_disc_actions.append(disc_pad)
                temp_disc_t_mask.append(disc_t_mask)

        if k in continuous_action_keys:
            cont_actions[k] = torch.stack(temp_cont_actions, dim=0).to(device)
            cont_actions_mask[k] = torch.stack(temp_cont_t_mask, dim=0).to(device)
        elif k in discrete_action_keys:
            disc_actions[k] = torch.stack(temp_disc_actions, dim=0).to(device)
            disc_actions_mask[k] = torch.stack(temp_disc_t_mask, dim=0).to(device)

    # ---------- core RL signals ----------
    rewards = [torch.tensor([t.reward for t in transitions], device=device)]
    rewards, rewards_mask = pad_temporal_others(rewards)
    dones = [torch.tensor(
        [t.terminated or t.truncated for t in transitions],
        dtype=torch.bool,
        device=device,
    )]
    dones, doens_mask = pad_temporal_others(dones)


    # ---------- extras ----------
    extras = {}
    extras_mask = {}
    if transitions[0].log_prob is not None:
        extras["log_prob"], extras_mask['log_prob'] = pad_temporal_others([torch.tensor([t.log_prob for t in transitions]).to(device)])
    if transitions[0].value is not None:
        extras["value"], extras_mask['value'] = pad_temporal_others(torch.tensor([t.value for t in transitions]).to(device))
    if transitions[0].hidden is not None:
        extras["hidden"], extras_mask['hidden'] = pad_temporal_others(torch.tensor([t.hidden for t in transitions]).to(device))
    if transitions[0].next_hidden is not None:
        extras["next_hidden"], extras_mask["next_hidden"] = pad_temporal_others(torch.tensor(
            [t.next_hidden for t in transitions]
        ).to(device))

    return Batch(
        box_obs=box_obs,
        seq_obs=seq_obs,
        seq_mask=seq_mask,
        next_seq_mask=next_seq_mask,
        cont_actions=cont_actions,
        cont_actions_mask=cont_actions_mask,
        disc_actions=disc_actions,
        disc_actions_mask=disc_actions_mask,
        rewards=rewards,
        reward_masks=rewards_mask,
        dones=dones,
        done_masks=doens_mask,
        next_box_obs=next_box_obs,
        next_seq_obs=next_seq_obs,
        extras=extras,
        extras_mask=extras_mask,
    )


def transitions_to_batch_sequence(batchSeqs, Lmax, device, box_obs=box_obs, seq_obs=seq_obs):

    keys = batchSeqs[0][0].states.keys()
    action_keys = batchSeqs[0][0].action.keys()

    # Keys being as types of observations and actions; values as Batch*TimeSteps*Dimensions

    # box-Type observations
    box_obs = {k: None for k in box_obs}
    next_box_obs = {k: None for k in box_obs}

    # Sequence-Type observations
    seq_obs = {k: None for k in seq_obs}
    seq_mask = {k: None for k in seq_obs}
    next_seq_obs = {k: None for k in seq_obs}
    next_seq_mask = {k: None for k in seq_obs}

    # actions
    disc_actions = {}
    cont_actions = {}
    cont_actions_mask = {}
    disc_actions_mask = {}

    for k in keys:
        temp_box_obs = [] 
        temp_next_box_obs = []
        temp_seqs_obs = []
        temp_seqs_mask = []
        temp_next_seq_obs = []
        temp_next_seq_mask = []
        temp_cont_actions = []
        temp_disc_actions = []

        for transitions in batchSeqs:
            # ---------- box obs ----------
            if k in box_obs:
                a = [torch.tensor(t.states[k]) for t in transitions]
                temp_box_obs.append(torch.stack(a))

                next_a = [torch.tensor(t.next_states[k]) for t in transitions]
                temp_next_box_obs.append(torch.stack(next_a))

            # ---------- seq obs ----------
            elif k in seq_obs:
                seq_list = [torch.tensor(t.states[k]).to(device) for t in transitions] # list of tensors (Seq_len, D) where Seq_len is variable, represeting the number of items in Sequence-Type obs
                padded_seqs, t_mask = pad_seq(seq_list, Lmax) # now Seq_len is padded to Lmax
                temp_seqs_obs.append(padded_seqs)
                temp_seqs_mask.append(t_mask)

                next_seqs = [torch.tensor(t.next_states[k]).to(device) for t in transitions]
                padded_n, mask_n = pad_seq(next_seqs, Lmax)
                temp_next_seq_obs.append(padded_n)
                temp_next_seq_mask.append(mask_n)
        
        if k in box_obs:
            box_obs[k], next_box_obs[k] = torch.stack(temp_box_obs).to(device), torch.stack(temp_next_box_obs).to(device)

        elif k in seq_obs:        
            seq_obs[k], seq_mask[k] = torch.stack(temp_seqs_obs, dim=0), torch.stack( temp_seqs_mask, dim=0) 
            next_seq_obs[k], next_seq_mask[k] = torch.stack(temp_next_seq_obs, dim=0), torch.stack(temp_next_seq_mask, dim=0)

    for k in action_keys:
        
        temp_cont_actions = [] 
        temp_cont_t_mask = []
        temp_disc_actions = []
        temp_disc_t_mask = [] 
        for transitions in batchSeqs:
            if k in continuous_action_keys:
                cont_pad, cont_mask = pad_temporal_others(
                    [torch.tensor(t.action[k]).squeeze() for t in transitions]
                )
                temp_cont_actions.append(cont_pad)
                temp_cont_t_mask.append(cont_mask)


            if k in discrete_action_keys:
                disc_pad, disc_t_mask = pad_temporal_others(
                    [torch.tensor(t.action[k]).squeeze() for t in transitions]
                )
                temp_disc_actions.append(disc_pad)
                temp_disc_t_mask.append(disc_t_mask)

        if k in continuous_action_keys:
            cont_actions[k] = torch.stack(temp_cont_actions, dim=0).to(device)
            cont_actions_mask[k] = torch.stack(temp_cont_t_mask, dim=0).to(device)
        elif k in discrete_action_keys:
            disc_actions[k] = torch.stack(temp_disc_actions, dim=0).to(device)
            disc_actions_mask[k] = torch.stack(temp_disc_t_mask, dim=0).to(device)

    # ---------- core RL signals ----------
    rewards = [torch.tensor([t.reward for t in transitions], device=device)]
    rewards, rewards_mask = pad_temporal_others(rewards)
    dones = [torch.tensor(
        [t.terminated or t.truncated for t in transitions],
        dtype=torch.bool,
        device=device,
    )]
    dones, doens_mask = pad_temporal_others(dones)


    # ---------- extras ----------
    extras = {}
    extras_mask = {}
    if transitions[0].log_prob is not None:
        extras["log_prob"], extras_mask['log_prob'] = pad_temporal_others([torch.tensor([t.log_prob for t in transitions]).to(device)])
    if transitions[0].value is not None:
        extras["value"], extras_mask['value'] = pad_temporal_others(torch.tensor([t.value for t in transitions]).to(device))
    if transitions[0].hidden is not None:
        extras["hidden"], extras_mask['hidden'] = pad_temporal_others(torch.tensor([t.hidden for t in transitions]).to(device))
    if transitions[0].next_hidden is not None:
        extras["next_hidden"], extras_mask["next_hidden"] = pad_temporal_others(torch.tensor(
            [t.next_hidden for t in transitions]
        ).to(device))

    return Batch(
        box_obs=box_obs,
        seq_obs=seq_obs,
        seq_mask=seq_mask,
        next_seq_mask=next_seq_mask,
        cont_actions=cont_actions,
        cont_actions_mask=cont_actions_mask,
        disc_actions=disc_actions,
        disc_actions_mask=disc_actions_mask,
        rewards=rewards,
        reward_masks=rewards_mask,
        dones=dones,
        done_masks=doens_mask,
        next_box_obs=next_box_obs,
        next_seq_obs=next_seq_obs,
        extras=extras,
        extras_mask=extras_mask,
    )
