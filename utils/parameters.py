import logging
LOGGING_LEVEL = logging.INFO
DRAG_MODE = 'X'
DRAG_MODE = 'XY'
NUM_HORIZON_POINTS = 21    # includes the last column
jpgDir = 'jpgDir'
maskDir = 'maskDir'

# https://www.webucator.com/article/python-color-constants-module/
color = {
    'Violet':	'#9400D3',
    'Indigo':	'#4B0082',
    'Blue':	    '#0000FF',
    'Aqua':	    '#00FFFF',
    'Green':	'#00FF00',
    'Yellow':	'#FFFF00',
    'Orange':	'#FF7F00',
    'DarkOrange': '#CD6600',
    'Red':	    '#FF0000',
    'Black':    '#FF0000',
    'White':    '#FFFFFF',
}

labels = {'Plane': color['Orange'],
    'Bird': color['Violet'],
    'Hang glider': color['Indigo'],
    'Paraglider': color['Aqua'],
    'Car': color['Blue'],
    'Boat': color['Green'],
    'Person': color['White'],
    'Horizon': color['Red'],
    }
