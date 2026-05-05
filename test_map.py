import pygame, sys, os

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
        "adjacent": ["Northwest Territory", "Ontario", "Quebec", "Iceland"],
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
        "adjacent": ["Venezuela", "Peru", "Argentina", "North Africa"],
    },
    "Argentina": {
        "polygon": [(480,960),(520,890),(620,940),(650,960),(560,960)],
        "center": (555,935),
        "continent": "South America",
        "adjacent": ["Peru", "Brazil"],
    },
}

WIN_W, WIN_H = 800, 920
SPRITE_W = 900
SPRITE_H = 1100

all_points = [p for d in TERRITORIES.values() for p in d["polygon"]]
max_x = max(p[0] for p in all_points)
max_y = max(p[1] for p in all_points)
min_x = min(p[0] for p in all_points)
min_y = min(p[1] for p in all_points)

PADDING = 10
scale_x = (WIN_W - PADDING * 2) / (max_x - min_x)
scale_y = (WIN_H - PADDING * 2) / (max_y - min_y)
SCALE = min(scale_x, scale_y)

def scale_point(p):
    return (int((p[0] - min_x) * SCALE) + PADDING,
            int((p[1] - min_y) * SCALE) + PADDING)

def scale_center(center):
    return (int((center[0] - min_x) * SCALE) + PADDING,
            int((center[1] - min_y) * SCALE) + PADDING)

pygame.init()
screen = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption("Map Preview")
font = pygame.font.SysFont("Arial", 13, bold=True)
clock = pygame.time.Clock()

# Load and scale all sprites
sprites = {}

def sprite_scale_center(center):
    cx, cy = center
    return (int(cx * WIN_W / SPRITE_W), int(cy * WIN_H / SPRITE_H))

for name in TERRITORIES:
    filename = name.lower().replace(" ", "_") + ".png"
    path = os.path.join("assets", filename)
    raw = pygame.image.load(path).convert_alpha()
    scaled = pygame.transform.scale(raw, (WIN_W, WIN_H))
    sprites[name] = scaled

while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.fill((180, 210, 230))

    # Blit all sprites
    for name in TERRITORIES:
        screen.blit(sprites[name], (0, 0))

    # Draw labels once using sprite scaling
    for name, data in TERRITORIES.items():
        cx, cy = sprite_scale_center(data["center"])
        label = font.render(name, True, (20, 20, 20))
        screen.blit(label, (cx - label.get_width()//2, cy - label.get_height()//2))

    pygame.display.flip()
    clock.tick(30)