from __future__ import print_function
import click
import itertools
import numpy as np
import pickle as P

@click.command()
@click.option('--event',
    'event_data',
    type=click.File('r'),
    prompt='Events file',
    help='Event data file.')
@click.option('--depth',
    'depth',
    default=1,
    type=int,
    help='Number of events in context in Markov Model.')
@click.option('--save',
    'save_file',
    prompt='Save file',
    type=click.File('wb'),
    help='File to save the results in.')
def run(event_data, depth, save_file):
    event_label_set = set()
    transition_count = {}
    init_state_count = {}
    for row_idx, event_row in enumerate(event_data):
        event_labels = [int(x) for x in event_row.split()]
        if len(event_labels) < depth:
            click.echo('Too few events in row {}. Skipping.'.format(row_idx))
            continue

        cur_state = tuple(event_labels[:depth])
        init_state_count[cur_state] = init_state_count.get(cur_state, 0) + 1

        for event_label in event_labels[depth:]:
            event_label_set.add(event_label)

            if cur_state not in transition_count:
                transition_count[cur_state] = {}

            transition_count[cur_state][event_label] = transition_count[cur_state].get(event_label, 0) + 1
            cur_state = cur_state[1:] + (event_label,)

    # Formulate and solve the problem here.
    num_labels = len(event_label_set)
    labels = sorted(list(event_label_set))
    states = tuple(itertools.product(labels, repeat=depth)) # Will be sorted

    # Performing MLE estimation for Markov Chains
    total_init_states = 1.0 * sum(init_state_count.values())
    init_state_probs = np.array([init_state_count.get(state, 0.0) / total_init_states
                                 for state in states])

    trans_probs = np.zeros((len(states), num_labels))

    for state_idx, state in enumerate(states):
        if state not in transition_count:
            click.echo('State {} was not observed at all!'.format(state))
            continue

        transitions_from_state = transition_count[state]
        total_transitions_from_state = sum(transitions_from_state.values())
        for target_idx, target_label in enumerate(labels):
            if target_label in transitions_from_state:
                trans_probs[state_idx, target_idx] = transitions_from_state[target_label] / total_transitions_from_state

    state = {
        'trans_prob': trans_probs,
        'init_state_probs': init_state_probs,
        'states': states,
        'labels': labels
    }
    P.dump(state, save_file)
    click.echo('Done.')
