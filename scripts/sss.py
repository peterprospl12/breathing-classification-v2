import pygame
import os

# Ustawienie backendu SDL na x11
os.environ["SDL_VIDEODRIVER"] = "x11"

try:
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    print("Pygame działa poprawnie.")
    pygame.quit()
except pygame.error as e:
    print(f"Pygame błąd: {e}")
