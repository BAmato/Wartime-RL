import pygame, sys, os
from env.map_config import TERRITORIES, TERRITORY_COLORS, WIN_W, WIN_H, SPRITE_W, SPRITE_H


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