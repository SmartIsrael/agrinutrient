import numpy as np
import pygame
from pygame import gfxdraw
import math

class HydroponicRenderer:
    """
    Renderer for the hydroponic environment using Pygame
    (simpler than OpenGL but still provides good visualization)
    """
    
    def __init__(self, render_mode='human'):
        self.render_mode = render_mode
        self.screen_width = 800
        self.screen_height = 600
        self.screen = None
        self.clock = None
        self.isopen = True
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)
        self.CYAN = (0, 255, 255)
        self.MAGENTA = (255, 0, 255)
        self.GRAY = (200, 200, 200)
        self.LIGHT_BLUE = (173, 216, 230)
        
        # Plant growth images (would be replaced with actual images in a real implementation)
        self.growth_colors = {
            0: self.GREEN,       # Seedling (light green)
            1: (0, 180, 0),      # Vegetative (medium green)
            2: (0, 150, 0),      # Flowering (darker green with hint of color)
            3: (0, 120, 0)       # Fruiting (darkest green)
        }
    
    def render(self, ph, ec, temperature, growth_stage, water_cycle):
        if self.screen is None:
            pygame.init()
            if self.render_mode == 'human':
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption("Hydroponic Environment")
            else:  # rgb_array
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Fill background
        self.screen.fill(self.WHITE)
        
        # Draw water reservoir
        pygame.draw.rect(self.screen, self.LIGHT_BLUE, (50, 400, 700, 150))
        
        # Draw hydroponic system structure
        pygame.draw.rect(self.screen, self.GRAY, (100, 350, 600, 50))  # Growing platform
        
        # Draw plants based on growth stage
        plant_positions = [(150 + i * 100, 350) for i in range(6)]
        
        for pos in plant_positions:
            self._draw_plant(pos[0], pos[1], growth_stage)
        
        # Draw pipes
        pygame.draw.rect(self.screen, self.GRAY, (400, 350, 10, 50))  # Vertical pipe
        pygame.draw.rect(self.screen, self.GRAY, (300, 550, 200, 10))  # Horizontal pipe
        
        # Draw pump
        pygame.draw.rect(self.screen, (100, 100, 100), (380, 560, 40, 30))
        
        # Draw water cycle indicator
        self._draw_water_cycle(700, 100, water_cycle)
        
        # Draw metrics
        self._draw_metrics(50, 50, ph, ec, temperature, growth_stage)
        
        # Add title
        font = pygame.font.SysFont(None, 36)
        title = font.render("Hydroponic Nutrient Optimization", True, self.BLACK)
        self.screen.blit(title, (self.screen_width // 2 - title.get_width() // 2, 10))
        
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(30)
        
        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
    
    def _draw_plant(self, x, y, growth_stage):
        # Base (pot)
        pygame.draw.rect(self.screen, (139, 69, 19), (x - 15, y, 30, 20))
        
        # Plant stem
        pygame.draw.rect(self.screen, (0, 100, 0), (x - 2, y - 20 * (growth_stage + 1), 4, 20 * (growth_stage + 1)))
        
        # Leaves/growth based on growth stage
        if growth_stage >= 0:  # Seedling
            pygame.draw.circle(self.screen, self.growth_colors[0], (x, y - 20), 5)
        
        if growth_stage >= 1:  # Vegetative
            pygame.draw.circle(self.screen, self.growth_colors[1], (x - 10, y - 30), 8)
            pygame.draw.circle(self.screen, self.growth_colors[1], (x + 10, y - 30), 8)
        
        if growth_stage >= 2:  # Flowering
            pygame.draw.circle(self.screen, self.growth_colors[2], (x - 12, y - 45), 10)
            pygame.draw.circle(self.screen, self.growth_colors[2], (x + 12, y - 45), 10)
            pygame.draw.circle(self.screen, (255, 255, 200), (x, y - 55), 5)  # Flower
        
        if growth_stage >= 3:  # Fruiting
            pygame.draw.circle(self.screen, self.growth_colors[3], (x - 15, y - 60), 12)
            pygame.draw.circle(self.screen, self.growth_colors[3], (x + 15, y - 60), 12)
            pygame.draw.circle(self.screen, (255, 0, 0), (x - 10, y - 70), 7)  # Fruit
            pygame.draw.circle(self.screen, (255, 0, 0), (x + 10, y - 70), 7)  # Fruit
    
    def _draw_metrics(self, x, y, ph, ec, temperature, growth_stage):
        font = pygame.font.SysFont(None, 24)
        
        # pH meter
        pygame.draw.rect(self.screen, self.BLACK, (x, y, 150, 20), 1)
        ph_color = self.GREEN if 5.5 <= ph <= 7.0 else self.RED
        ph_width = min(150, max(0, (ph / 14.0) * 150))
        pygame.draw.rect(self.screen, ph_color, (x, y, ph_width, 20))
        ph_text = font.render(f"pH: {ph:.1f}", True, self.BLACK)
        self.screen.blit(ph_text, (x + 160, y))
        
        # EC meter
        pygame.draw.rect(self.screen, self.BLACK, (x, y + 40, 150, 20), 1)
        ec_color = self.GREEN if 0.8 <= ec <= 3.0 else self.RED
        ec_width = min(150, max(0, (ec / 5.0) * 150))
        pygame.draw.rect(self.screen, ec_color, (x, y + 40, ec_width, 20))
        ec_text = font.render(f"EC: {ec:.1f} mS/cm", True, self.BLACK)
        self.screen.blit(ec_text, (x + 160, y + 40))
        
        # Temperature meter
        pygame.draw.rect(self.screen, self.BLACK, (x, y + 80, 150, 20), 1)
        temp_color = self.GREEN if 18.0 <= temperature <= 28.0 else self.RED
        temp_width = min(150, max(0, ((temperature - 10) / 30) * 150))
        pygame.draw.rect(self.screen, temp_color, (x, y + 80, temp_width, 20))
        temp_text = font.render(f"Temp: {temperature:.1f}°C", True, self.BLACK)
        self.screen.blit(temp_text, (x + 160, y + 80))
        
        # Growth stage
        stage_names = ["Seedling", "Vegetative", "Flowering", "Fruiting"]
        stage_text = font.render(f"Growth Stage: {stage_names[growth_stage]}", True, self.BLACK)
        self.screen.blit(stage_text, (x, y + 120))
    
    def _draw_water_cycle(self, x, y, water_cycle):
        font = pygame.font.SysFont(None, 24)
        cycle_text = font.render(f"Water Cycle: {water_cycle} hours", True, self.BLACK)
        self.screen.blit(cycle_text, (x, y))
        
        # Visual representation of water cycle
        center = (x + 30, y + 50)
        radius = 20
        
        # Draw circle
        pygame.draw.circle(self.screen, self.BLACK, center, radius, 1)
        
        # Draw "clock hand" based on water cycle (longer cycle = slower hand)
        angle = 2 * math.pi * (((water_cycle % 12) / 12))
        end_x = center[0] + radius * math.sin(angle)
        end_y = center[1] - radius * math.cos(angle)
        pygame.draw.line(self.screen, self.BLUE, center, (end_x, end_y), 2)
    
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.isopen = False