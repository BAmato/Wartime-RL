# Territory map data
TERRITORIES = {
    "Alaska": {
        "polygon": [(80,80),(260,70),(300,160),(300,260),(240,230),(90,220)],
        "center": (185,165),
        "continent": "North America",
        "adjacent": ["Northwest Territory", "Alberta", "Kamchatka"],
    },
    "Northwest Territory": {
        "polygon": [(260,70),(520,70),(560,180),(470,260),(300,260),(240,230),(300,160)],
        "center": (395,165),
        "continent": "North America",
        "adjacent": ["Alaska", "Alberta", "Ontario", "Greenland"],
    },
    "Greenland": {
        "polygon": [(520,70),(820,70),(900,170),(760,230),(650,240),(560,180)],
        "center": (705,145),
        "continent": "North America",
        "adjacent": ["Northwest Territory", "Ontario", "Quebec"],
    },
    "Alberta": {
        "polygon": [(240,230),(300,260),(470,260),(460,390),(270,395)],
        "center": (365,320),
        "continent": "North America",
        "adjacent": ["Alaska", "Northwest Territory", "Ontario", "Western United States"],
    },
    "Ontario": {
        "polygon": [(470,260),(560,180),(650,240),(660,380),(560,450),(460,390)],
        "center": (560,325),
        "continent": "North America",
        "adjacent": ["Northwest Territory", "Alberta", "Greenland", "Quebec", "Eastern United States", "Western United States"],
    },
    "Quebec": {
        "polygon": [(650,240),(760,230),(820,360),(740,450),(660,380)],
        "center": (725,335),
        "continent": "North America",
        "adjacent": ["Greenland", "Ontario", "Eastern United States"],
    },
    "Western United States": {
        "polygon": [(270,395),(460,390),(560,450),(540,560),(380,570),(250,520)],
        "center": (405,485),
        "continent": "North America",
        "adjacent": ["Alberta", "Ontario", "Eastern United States", "Central America"],
    },
    "Eastern United States": {
        "polygon": [(560,450),(660,380),(740,450),(710,580),(590,640),(540,560)],
        "center": (640,515),
        "continent": "North America",
        "adjacent": ["Ontario", "Quebec", "Western United States", "Central America"],
    },
    "Central America": {
        "polygon": [(250,520),(380,570),(540,560),(590,640),(500,700),(330,660)],
        "center": (430,620),
        "continent": "North America",
        "adjacent": ["Western United States", "Eastern United States", "Venezuela"],
    },
    "Venezuela": {
        "polygon": [(330,660),(500,700),(610,760),(500,820),(360,790)],
        "center": (465,745),
        "continent": "South America",
        "adjacent": ["Central America", "Peru", "Brazil"],
    },
    "Peru": {
        "polygon": [(360,790),(500,820),(520,890),(480,960),(330,930),(300,850)],
        "center": (420,875),
        "continent": "South America",
        "adjacent": ["Venezuela", "Brazil", "Argentina"],
    },
    "Brazil": {
        "polygon": [(500,820),(610,760),(760,800),(800,930),(620,940),(520,890)],
        "center": (645,850),
        "continent": "South America",
        "adjacent": ["Venezuela", "Peru", "Argentina"],
    },
    "Argentina": {
        "polygon": [(480,960),(520,890),(620,940),(650,960),(560,960)],
        "center": (555,935),
        "continent": "South America",
        "adjacent": ["Peru", "Brazil"],
    },
}

CONTINENTS = {
    "North America": {
        "territories": ["Alaska", "Northwest Territory", "Greenland", "Alberta",
                        "Ontario", "Quebec", "Western United States",
                        "Eastern United States", "Central America"],
        "bonus_armies": 5,
    },
    "South America": {
        "territories": ["Venezuela", "Peru", "Brazil", "Argentina"],
        "bonus_armies": 2,
    },
}

# Pre-compute all valid attack pairs
ATTACK_PAIRS = [
    (src, tgt)
    for src, data in TERRITORIES.items()
    for tgt in data["adjacent"]
    if tgt in TERRITORIES  # only pairs within our map
]

# Sprite colors per territory
TERRITORY_COLORS = {
    "Alaska": (150,190,230), "Northwest Territory": (130,175,215),
    "Greenland": (170,205,240), "Alberta": (145,185,225),
    "Ontario": (125,170,210), "Quebec": (160,200,235),
    "Western United States": (210,185,100), "Eastern United States": (195,170,85),
    "Central America": (220,195,110), "Venezuela": (210,140,100),
    "Brazil": (195,125,85), "Peru": (220,150,110), "Argentina": (205,135,90)
}

OWNER_TINT = {
    "agent":   (100, 180, 100),  # green
    "enemy":   (200, 80,  80),   # red
    "neutral": None              # use default territory color
}

# Sprite canvas dimensions
SPRITE_W = 900
SPRITE_H = 1100

# Window dimensions
WIN_W = 800
WIN_H = 920