def split_trial(file, trial_start, trial_end, window_size, window_offset):
    windows_coordinates = []
    window_start = trial_start

    while True:
        window_end = window_start + window_size - 1

        if window_end == trial_end:
            windows_coordinates.append([window_start, window_end])
            break

        if window_end > trial_end:
            windows_coordinates[-1][1] = trial_end
            break

        windows_coordinates.append([window_start, window_end])
        window_start += window_offset

    for coordinate in windows_coordinates:
        print(f'{coordinate[0]} {coordinate[1]}', file = file)
