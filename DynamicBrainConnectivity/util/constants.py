
import plotly.express as px


SUBJECT_FILE_PREFIX = 'Dots_30_'
CSV_FILE_EXTENSION = '.csv'
SUBJECT_FILE_EXTENSION = '.bin'
TRIAL_DATA = '-Trial-Data'
SUBJECT_FILE_CHANNEL = '-Ch'
SUBJECT_FILE_EVENT_CODES = '-Event-Codes'
SUBJECT_FILE_EVENT_TIMESTAMPS = '-Event-Timestamps'
PARSED_DATA = 'RawDataFilter'

EVENT_CODES_FILTER = [1, 2, 3, 129]

NUMBER_OF_CHANNELS = 128
NUMBER_OF_TRIALS = 210
DIVISION_LENGTH = 600

TEXT_FILE_EXTENSION = ".txt"
IMAGE_FILE_EXTENSION = ".png"
HTML_FILE_EXTENSION = ".html"

STIMULUS_OUTPUT_SIZE = 6
RESPONSE_OUTPUT_SIZE = 3

COLOR_LIST_DISTRIBUTION_PLOTS = [
    'blueviolet',
    'coral',
    'forestgreen',
    'red',
    'fuchsia',
    'pink',
    'rosybrown',
]

COLOR_LIST_CHANNELS = []
COLOR_LIST_CHANNELS.extend(px.colors.qualitative.Plotly)
COLOR_LIST_CHANNELS.extend(px.colors.qualitative.D3)
COLOR_LIST_CHANNELS.extend(px.colors.qualitative.G10)
COLOR_LIST_CHANNELS.extend(px.colors.qualitative.T10)
COLOR_LIST_CHANNELS.extend(px.colors.qualitative.Set1)
COLOR_LIST_CHANNELS.extend(px.colors.qualitative.Vivid)
COLOR_LIST_CHANNELS.extend(px.colors.qualitative.Safe)
COLOR_LIST_CHANNELS.extend(px.colors.qualitative.Prism)
COLOR_LIST_CHANNELS.extend(px.colors.qualitative.Pastel)
COLOR_LIST_CHANNELS.extend(px.colors.qualitative.Bold)
COLOR_LIST_CHANNELS.extend(px.colors.qualitative.Dark24)
COLOR_LIST_CHANNELS.extend(px.colors.qualitative.Light24)

GRAPH_WAVENET_LOADER_OPTIONS = [
    'Trial',
    'Window'
]

GRAPH_WAVENET_LOSS_TYPES = [
    'MAE'
]

CHANNELS_DICT = {
    0: 'A1',
    1: 'A2',
    2: 'A3',
    3: 'A4',
    4: 'A5',
    5: 'A6',
    6: 'A7',
    7: 'A8',
    8: 'A9',
    9: 'A10',
    10: 'A11',
    11: 'A12',
    12: 'A13',
    13: 'A14',
    14: 'A15',
    15: 'A16',
    16: 'A17',
    17: 'A18',
    18: 'A19',
    19: 'A20',
    20: 'A21',
    21: 'A22',
    22: 'A23',
    23: 'A24',
    24: 'A25',
    25: 'A26',
    26: 'A27',
    27: 'A28',
    28: 'A29',
    29: 'A30',
    30: 'A31',
    31: 'A32',
    32: 'B1',
    33: 'B2',
    34: 'B3',
    35: 'B4',
    36: 'B5',
    37: 'B6',
    38: 'B7',
    39: 'B8',
    40: 'B9',
    41: 'B10',
    42: 'B11',
    43: 'B12',
    44: 'B13',
    45: 'B14',
    46: 'B15',
    47: 'B16',
    48: 'B17',
    49: 'B18',
    50: 'B19',
    51: 'B20',
    52: 'B21',
    53: 'B22',
    54: 'B23',
    55: 'B24',
    56: 'B25',
    57: 'B26',
    58: 'B27',
    59: 'B28',
    60: 'B29',
    61: 'B30',
    62: 'B31',
    63: 'B32',
    64: 'C1',
    65: 'C2',
    66: 'C3',
    67: 'C4',
    68: 'C5',
    69: 'C6',
    70: 'C7',
    71: 'C8',
    72: 'C9',
    73: 'C10',
    74: 'C11',
    75: 'C12',
    76: 'C13',
    77: 'C14',
    78: 'C15',
    79: 'C16',
    80: 'C17',
    81: 'C18',
    82: 'C19',
    83: 'C20',
    84: 'C21',
    85: 'C22',
    86: 'C23',
    87: 'C24',
    88: 'C25',
    89: 'C26',
    90: 'C27',
    91: 'C28',
    92: 'C29',
    93: 'C30',
    94: 'C31',
    95: 'C32',
    96: 'D1',
    97: 'D2',
    98: 'D3',
    99: 'D4',
    100: 'D5',
    101: 'D6',
    102: 'D7',
    103: 'D8',
    104: 'D9',
    105: 'D10',
    106: 'D11',
    107: 'D12',
    108: 'D13',
    109: 'D14',
    110: 'D15',
    111: 'D16',
    112: 'D17',
    113: 'D18',
    114: 'D19',
    115: 'D20',
    116: 'D21',
    117: 'D22',
    118: 'D23',
    119: 'D24',
    120: 'D25',
    121: 'D26',
    122: 'D27',
    123: 'D28',
    124: 'D29',
    125: 'D30',
    126: 'D31',
    127: 'D32',
}

# channels_parietal = ['A 1', 'A 2', 'B 1', 'C 1', 'D 1', 'D 15', 'A 3', 'B 2', 'B 20', 'C 2', 'C 23', 'D 2', 'D 14',
#                      'D 16', 'A 4', 'B 19', 'B 21', 'B 32', 'C 11', 'C 22', 'C 24', 'D 13', 'D 18', 'D 17']
# channels_frontal = ['C 12', 'C 13', 'C 14', 'C 15', 'C 16', 'C 17', 'C 18', 'C 19', 'C 20', 'C 21', 'C 25', 'C 26',
#                     'C 27', 'C 28', 'C 29']
# channels_frontal_left = ['D 3', 'D 4', 'D 5', 'D 6', 'D 7', 'C 30', 'C 31', 'C 32']
# channels_frontal_right = ['C 3', 'C 4', 'C 5', 'C 6', 'C 7', 'C 8', 'C 9', 'C 10']
# channels_occipital = ['A 5', 'A 13', 'A 14', 'A 15', 'A 16', 'A 17', 'A 18', 'A 19', 'A 20', 'A 21', 'A 22', 'A 23',
#                       'A 24', 'A 25', 'A 26', 'A 27', 'A 28', 'A 29', 'A 30', 'A 31', 'A 32']
# channels_occipital_left = ['A 6', 'A 7', 'A 8', 'A 9', 'A 10', 'A 11', 'A 12', 'D 29', 'D 30', 'D 31', 'D 32']
# channels_occipital_right = ['B 3', 'B 4', 'B 5', 'B 6', 'B 7', 'B 8', 'B 9', 'B 10', 'B 11', 'B 12', 'B 13']
# channels_temporal_left = ['D 8', 'D 9', 'D 10', 'D 11', 'D 12', 'D 19', 'D 20', 'D 21', 'D 22', 'D 23', 'D 24', 'D 25',
#                           'D 26', 'D 27', 'D 28']
# channels_temporal_right = ['B 14', 'B 15', 'B 16', 'B 17', 'B 18', 'B 22', 'B 23', 'B 24', 'B 25', 'B 26', 'B 27',
#                            'B 28', 'B 29', 'B 30', 'B 31']

CHANNELS_PARIETAL = [0, 1, 32, 64, 96, 110, 2, 33, 51, 65, 86, 97, 109, 111, 3, 50, 52, 63, 74, 85, 87, 108, 113, 112]
CHANNELS_FRONTAL = [75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 88, 89, 90, 91, 92]
CHANNELS_FRONTAL_LEFT = [98, 99, 100, 101, 102, 93, 94, 95]
CHANNELS_FRONTAL_RIGHT = [66, 67, 68, 69, 70, 71, 72, 73]
CHANNELS_OCCIPITAL = [4, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
CHANNELS_OCCIPITAL_LEFT = [5, 6, 7, 8, 9, 10, 11, 124, 125, 126, 127]
CHANNELS_OCCIPITAL_RIGHT = [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
CHANNELS_TEMPORAL_LEFT = [103, 104, 105, 106, 107, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123]
CHANNELS_TEMPORAL_RIGHT = [45, 46, 47, 48, 49, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]

CHANNELS_PLACEMENT = [
    CHANNELS_PARIETAL,
    CHANNELS_OCCIPITAL_LEFT,
    CHANNELS_OCCIPITAL,
    CHANNELS_OCCIPITAL_RIGHT,
    CHANNELS_TEMPORAL_RIGHT,
    CHANNELS_FRONTAL_RIGHT,
    CHANNELS_FRONTAL,
    CHANNELS_FRONTAL_LEFT,
    CHANNELS_TEMPORAL_LEFT
]

CHANNELS_PLACEMENT_LABEL = [
    'PAR',
    'OCC\nLEFT',
    'OCC',
    'OCC\nRIGHT',
    'TEMP\nRIGHT',
    'FRONT\nRIGHT',
    'FRONT',
    'FRONT\nLEFT',
    'TEMP\nLEFT'
]

INTERVAL_START = 0.0
INTERVAL_END = 1.0

CHANNELS_PLACEMENT_COORD = [[0, 0], [-3.5, -3.5], [0, -5], [3.5, -3.5], [5, 0], [3.5, 3.5], [0, 5], [-3.5, 3.5],
                            [-5, 0]]
CHANNELS_PLACEMENT_COORD_LABELS = [[0, 0.5], [-4, -4], [0, -5.5], [4.0, -4.0], [5.6, 0], [4, 4], [0, 5.5], [-4, 4],
                                   [-5.6, 0]]

NUMBER_OF_MAX_CLIQUES = 'number_of_max_cliques'
MAX_CLIQUE_LENGTH = 'max_clique_length'
LONGEST_COMPONET_LENGTH = 'longest_componet_length'
STRONGLY_CONNECTED_COMPONENTS = 'strongly_connected_components'
MSA_WEIGHT = 'MSA weight'
AVG_MAX_WEIGHT_PATH = 'avg_max_weight_path'
AVG_SHORTEST_PATH = 'avg_shortest_path'
AVG_NR_OF_TRIANGLES = 'avg_nr_of_triangles'
TRANSITIVITY = "transitivity"
UNDIRECTED_AVG_CLUSTERING = "undirected_avg_clustering"
DIRECTED_AVG_CLUSTERING = "directed_avg_clustering"
DIRECTED_WEIGHTED_AVG_CLUSTERING = "directed_weighted_avg_clustering"
UNDIRECTED_AVG_SQ_CLUSTERING = "undirected_avg_sq_clustering"
DIRECTED_AVG_SQ_CLUSTERING = "directed_avg_sq_clustering"
AVG_DEGREE_CENTRALITY = 'avg_degree_centrality'
AVG_IN_DEGREE_CENTRALITY = 'avg_in_degree_centrality'
AVG_OUT_DEGREE_CENTRALITY = 'avg_out_degree_centrality'
AVG_UNWEIGHTED_BETWEENES_CENTRALITY = 'avg_unweighted_betweenes_centrality'
AVG_WEIGHTED_BETWEENES_CENTRALITY = 'avg_weighted_betweenes_centrality'

TRIALS_FOR_STIMULUS = {
    'poseta / geanta (de dama)': [1, 32, 83, 107, 145, 158, 186],
    'topor / secure': [2, 50, 70, 98, 144, 171, 191],
    'oala / cratita': [3, 55, 86, 120, 147, 160, 200],
    'elicopter': [4, 42, 78, 100, 134, 172, 193],
    'urs (polar)': [5, 37, 87, 94, 124, 170, 182],
    'palarie': [6, 44, 84, 101, 127, 155, 195],
    'foarfece': [7, 60, 79, 117, 143, 166, 199],
    'banana': [8, 52, 71, 111, 150, 156, 194],
    'lampa / veioza': [9, 31, 75, 115, 137, 157, 183],
    'chitara (electrica)': [10, 35, 77, 114, 139, 167, 196],
    'masina': [11, 57, 62, 103, 146, 169, 204],
    'vaca': [12, 46, 64, 95, 130, 173, 207],
    'furculita': [13, 48, 74, 91, 128, 153, 206],
    'cerb': [14, 49, 66, 118, 132, 164, 190],
    'pantaloni (scurti)': [15, 36, 80, 97, 135, 177, 185],
    'scaun': [16, 53, 81, 113, 131, 161, 198],
    'peste': [17, 38, 90, 106, 123, 162, 188],
    'caine / catel': [18, 33, 68, 92, 129, 168, 192],
    'sticla': [19, 54, 69, 96, 121, 179, 205],
    'pistol': [20, 59, 65, 102, 133, 174, 201],
    'bicicleta': [21, 40, 82, 116, 141, 163, 197],
    'cal': [22, 45, 76, 99, 140, 180, 184],
    'elefant': [23, 43, 89, 108, 149, 175, 202],
    'iepure': [24, 56, 61, 104, 142, 152, 189],
    'pahar / cupa': [25, 47, 73, 112, 138, 165, 181],
    'masa': [26, 39, 88, 110, 125, 176, 208],
    'umbrela': [27, 51, 67, 109, 126, 151, 210],
    'fluture': [28, 34, 63, 119, 122, 154, 187],
    'girafa': [29, 58, 72, 93, 136, 159, 209],
    'pian': [30, 41, 85, 105, 148, 178, 203]
}

RESPONSE_FOR_STIMULUS = {
    'poseta / geanta (de dama)': [3, 3, 3, 2, 2, 2, 1],
    'topor / secure': [3, 3, 2, 2, 1, 1, 1],
    'oala / cratita': [3, 3, 3, 2, 2, 1, 1],
    'elicopter': [3, 3, 3, 2, 1, 1, 1],
    'urs (polar)': [3, 3, 3, 2, 2, 2, 2],
    'palarie': [3, 3, 3, 2, 1, 1, 1],
    'foarfece': [3, 3, 3, 3, 2, 2, 1],
    'banana': [3, 3, 3, 2, 2, 1, 1],
    'lampa / veioza': [3, 3, 3, 3, 2, 2, 2],
    'chitara (electrica)': [3, 3, 2, 2, 1, 1, 1],
    'masina': [3, 3, 2, 2, 2, 1, 1],
    'vaca': [3, 3, 3, 2, 1, 1, 1],
    'furculita': [3, 3, 3, 2, 1, 1, 1],
    'cerb': [3, 3, 3, 1, 1, 1, 1],
    'pantaloni (scurti)': [3, 3, 3, 2, 2, 1, 1],
    'scaun': [3, 3, 3, 2, 1, 1, 1],
    'peste': [3, 3, 3, 2, 1, 1, 1],
    'caine / catel': [3, 3, 3, 2, 1, 1, 1],
    'sticla': [3, 3, 3, 3, 3, 2, 1],
    'pistol': [3, 3, 3, 2, 1, 1, 1],
    'bicicleta': [3, 3, 3, 3, 2, 2, 1],
    'cal': [3, 3, 3, 2, 2, 1, 1],
    'elefant': [3, 3, 3, 3, 2, 2, 1],
    'iepure': [3, 3, 3, 2, 1, 2, 1],
    'pahar / cupa': [3, 3, 3, 3, 2, 2, 2],
    'masa': [3, 3, 3, 2, 1, 1, 1],
    'umbrela': [3, 3, 3, 2, 1, 2, 1],
    'fluture': [3, 3, 3, 2, 1, 1, 1],
    'girafa': [3, 3, 3, 2, 2, 1, 1],
    'pian': [3, 3, 3, 3, 2, 2, 1]
}

STIMULUS_FOR_RESPONSE_SEQ = {
    '3332221': ['poseta / geanta (de dama)'],
    '3322111': ['topor / secure', 'chitara (electrica)'],
    '3332211': ['oala / cratita', 'banana', 'pantaloni (scurti)', 'cal', 'girafa'],
    '3332111': ['elicopter', 'palarie', 'vaca', 'furculita', 'scaun', 'peste',
                'caine / catel', 'pistol', 'masa', 'fluture'],
    '3332222': ['urs (polar)'],
    '3333221': ['foarfece', 'bicicleta', 'elefant', 'pian'],
    '3333222': ['lampa / veioza', 'pahar / cupa'],
    '3322211': ['masina'],
    '3331111': ['cerb'],
    '3333321': ['sticla'],
    '3332121': ['iepure', 'umbrela']
}
