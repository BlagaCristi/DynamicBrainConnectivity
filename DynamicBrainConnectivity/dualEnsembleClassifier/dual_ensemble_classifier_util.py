def convert_trial_dictionary_to_example(dictionary, window_size, window_offset, division_length):
    """
    Converts the trial to two examples using windows starting from the stimulus
    and going to the response.
    Number of windows and window offset (they overlap) is given.
    The windows are concatenated
    """

    stimulus_example = []
    response_example = []

    # starting index for the response windows (the leftmost window is the first)
    offset_response = division_length
    nr_windows = (division_length - window_size) // window_offset + 1
    if (division_length - window_size) % window_offset != 0:
        nr_windows += 1
    for window_index in range(0, nr_windows):
        # shift the window to the right with window_offset starting from 0
        offset_stimulus = window_offset * window_index

        # extend stimulus example with the new window
        stimulus_example.extend(
            dictionary['values'][offset_stimulus: min(division_length, offset_stimulus + window_size)]
        )

        # the windows are ordered in an increasing manner based on the timestamp
        response_example.extend(
            dictionary['values'][-(offset_response - offset_stimulus):][:window_size]
        )
    return {
        'stimulus': dictionary['stimulus'],
        'response': dictionary['response'],
        'stimulus_values': stimulus_example,
        'response_values': response_example,
        'g': dictionary['g']
    }
