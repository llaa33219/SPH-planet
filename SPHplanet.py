import sys
import math
import numpy as np
import pygame
from pygame.locals import *
from numba import njit
import time

# ============================
# Config
# ============================
class Config:
    WIDTH = 1000
    HEIGHT = 700
    SIDE_PANEL_WIDTH = 200
    FPS = 60

    BG_COLOR = (10, 10, 30)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    DT = 0.0005
    H = 15.0
    RHO0 = 1000.0
    K = 3.0e3
    MU = 10.0
    GRAVITY = 30.0

    FONT_NAME = 'malgungothic'
    FONT_SIZE = 18

    COLLISION_DISTANCE_FACTOR = 0.5
    COLLISION_IMPULSE = 20.0

# ============================
# SPH Kernels (Numba)
# ============================
@njit
def poly6_kernel(r, h):
    if 0 <= r <= h:
        factor = (315.0/(64.0*math.pi*(h**9)))
        return factor * (h**2 - r**2)**3
    else:
        return 0.0

@njit
def spiky_gradient(r_vec, h):
    r = math.sqrt(r_vec[0]**2 + r_vec[1]**2)
    if 0 < r <= h:
        factor = -45.0/(math.pi*(h**6))
        val = factor * (h-r)**2
        return val * (r_vec / r)
    else:
        return np.array([0.0, 0.0])

@njit
def viscosity_laplacian(r, h):
    if 0 <= r <= h:
        return 45.0/(math.pi*(h**6))*(h-r)
    else:
        return 0.0

@njit
def compute_densities(positions, masses, densities, H):
    n = positions.shape[0]
    for i in range(n):
        rho = 0.0
        p_i = positions[i]
        for j in range(n):
            r_vec = p_i - positions[j]
            r = math.sqrt(r_vec[0]**2 + r_vec[1]**2)
            rho += masses[j] * poly6_kernel(r, H)
        densities[i] = rho

@njit
def compute_pressures(densities, pressures, RHO0, K):
    n = densities.shape[0]
    for i in range(n):
        pressures[i] = K*(densities[i]-RHO0)

@njit
def compute_forces(positions, velocities, masses, densities, pressures, forces, MU, H, GRAVITY):
    n = positions.shape[0]
    for i in range(n):
        f = np.array([0.0, 0.0])
        p_i = positions[i]
        v_i = velocities[i]
        for j in range(n):
            if i == j:
                continue
            r_vec = p_i - positions[j]
            r = math.sqrt(r_vec[0]**2 + r_vec[1]**2)
            if 0 < r < H:
                p_term = (pressures[i] + pressures[j])/(2.0*densities[j])
                f += -p_term * spiky_gradient(r_vec, H) * masses[j]
                f += MU * (velocities[j]-v_i)*(masses[j]/densities[j])*viscosity_laplacian(r, H)
        f += np.array([0.0, GRAVITY])
        forces[i] = f

@njit
def integrate(positions, velocities, forces, densities, DT):
    n = positions.shape[0]
    for i in range(n):
        a = forces[i]/densities[i]
        velocities[i] += a*DT
        positions[i] += velocities[i]*DT

# ============================
# Planet Class
# ============================
class Planet:
    def __init__(self, name, radius, mass, init_vel, color):
        self.name = name
        self.radius = radius
        self.mass_total = mass
        self.init_vel = init_vel
        self.color = color
        self.positions = np.zeros((0,2), dtype=np.float64)
        self.velocities = np.zeros((0,2), dtype=np.float64)
        self.masses = np.zeros((0,), dtype=np.float64)
        self.colors = np.zeros((0,3), dtype=np.float64)
        self.densities = np.zeros((0,), dtype=np.float64)
        self.pressures = np.zeros((0,), dtype=np.float64)
        self.forces = np.zeros((0,2), dtype=np.float64)

    def spawn_particles(self, center, explosion=False):
        # 파티클 개수는 행성 반지름에 비례하게 설정
        N = max(150, int((self.radius*self.radius)/2))
        pos_list = []
        vel_list = []
        mass_list = []
        color_list = []
        for _ in range(N):
            theta = 2*math.pi*np.random.rand()
            r = self.radius * math.sqrt(np.random.rand())
            x = center[0] + r*math.cos(theta)
            y = center[1] + r*math.sin(theta)

            # 기본적으로 폭발 속도 = 0
            vx_explosion = 0.0
            vy_explosion = 0.0
            if explosion:
                # 필요시 폭발 : 랜덤 속도 부여
                explosion_speed = 5.0 + 5.0 * np.random.rand()
                vx_explosion = explosion_speed * math.cos(theta)
                vy_explosion = explosion_speed * math.sin(theta)

            # 행성 초기 속도 + 폭발 속도
            vx = self.init_vel[0] + vx_explosion
            vy = self.init_vel[1] + vy_explosion

            pos_list.append([x,y])
            vel_list.append([vx, vy])
            mass_list.append(self.mass_total/N)
            color_list.append(self.color)

        self.positions = np.array(pos_list, dtype=np.float64)
        self.velocities = np.array(vel_list, dtype=np.float64)
        self.masses = np.array(mass_list, dtype=np.float64)
        self.colors = np.array(color_list, dtype=np.float64)
        self.update_arrays()

    def update_arrays(self):
        n = self.positions.shape[0]
        self.densities = np.zeros((n,), dtype=np.float64)
        self.pressures = np.zeros((n,), dtype=np.float64)
        self.forces = np.zeros((n,2), dtype=np.float64)

    def integrate_sph(self, config):
        compute_densities(self.positions, self.masses, self.densities, config.H)
        compute_pressures(self.densities, self.pressures, config.RHO0, config.K)
        compute_forces(self.positions, self.velocities, self.masses, self.densities, self.pressures, self.forces,
                       config.MU, config.H, config.GRAVITY)
        integrate(self.positions, self.velocities, self.forces, self.densities, config.DT)

        # 경계 처리
        for i in range(self.positions.shape[0]):
            if self.positions[i,0] < 0:
                self.positions[i,0] = 0
                self.velocities[i,0] *= -0.5
            if self.positions[i,0] > config.WIDTH - config.SIDE_PANEL_WIDTH:
                self.positions[i,0] = config.WIDTH - config.SIDE_PANEL_WIDTH
                self.velocities[i,0] *= -0.5
            if self.positions[i,1] < 0:
                self.positions[i,1] = 0
                self.velocities[i,1] *= -0.5
            if self.positions[i,1] > config.HEIGHT:
                self.positions[i,1] = config.HEIGHT
                self.velocities[i,1] *= -0.5

    def render(self, screen):
        for i in range(self.positions.shape[0]):
            x, y = self.positions[i]
            c = self.colors[i]
            col = (int(c[0]), int(c[1]), int(c[2]))
            pygame.draw.circle(screen, col, (int(x), int(y)), 2)

    def get_mass(self):
        return np.sum(self.masses)

    def get_center(self):
        return np.mean(self.positions, axis=0) if self.positions.shape[0]>0 else np.array([0,0])

    def get_velocity_mean(self):
        if self.velocities.shape[0] > 0:
            return np.mean(self.velocities, axis=0)
        return np.array([0,0])

    def apply_collision_impulse(self, direction, impulse):
        # 행성 전체 파티클에 충돌 반발 속도를 더해준다.
        self.velocities += direction * (impulse/100.0)

# ============================
# GameState Class
# ============================
class GameState:
    def __init__(self, config):
        self.config = config
        self.planets = []
        self.paused = False
        self.selected_planet_type = None
        try:
            self.font = pygame.font.SysFont(config.FONT_NAME, config.FONT_SIZE)
        except:
            self.font = pygame.font.SysFont(None, config.FONT_SIZE)
        self.start_time = time.time()
        self.collision_count = 0

    def add_planet(self, planet_type, spawn_pos, explosion=False):
        p = Planet(planet_type['name'], planet_type['radius'], planet_type['mass'], planet_type['init_vel'], planet_type['color'])
        # 필요시에만 explosion=True로 폭발 효과 사용
        p.spawn_particles(spawn_pos, explosion=explosion)
        self.planets.append(p)

    def update(self):
        if not self.paused:
            for p in self.planets:
                p.integrate_sph(self.config)

            # 충돌 체크(단순): 행성 중심 거리로 판단
            for i in range(len(self.planets)):
                for j in range(i+1, len(self.planets)):
                    pi = self.planets[i]
                    pj = self.planets[j]
                    if pi.positions.shape[0] > 0 and pj.positions.shape[0] > 0:
                        ci = pi.get_center()
                        cj = pj.get_center()
                        dist = np.linalg.norm(ci - cj)
                        # 충돌 조건: 중심 거리 < (합의 반경) * FACTOR
                        if dist < (pi.radius + pj.radius)*self.config.COLLISION_DISTANCE_FACTOR:
                            # 충돌 발생 → 간단한 반발 처리
                            self.collision_count += 1
                            normal = (ci - cj)
                            if np.linalg.norm(normal) < 1e-8:
                                normal = np.array([1.0,0.0])
                            else:
                                normal = normal / np.linalg.norm(normal)
                            # 양쪽 행성 반대 방향으로 속도 변화
                            pi.apply_collision_impulse(normal, self.config.COLLISION_IMPULSE)
                            pj.apply_collision_impulse(-normal, self.config.COLLISION_IMPULSE)

    def render_ui(self, screen):
        title_surf = self.font.render("SPH 행성 충돌 시뮬레이션", True, self.config.WHITE)
        screen.blit(title_surf, (10, 10))

        info_surf = self.font.render(f"행성 수: {len(self.planets)} | 충돌 횟수: {self.collision_count}", True, self.config.WHITE)
        screen.blit(info_surf, (10, 40))

        if self.selected_planet_type is not None:
            sel_surf = self.font.render("선택된 행성: "+self.selected_planet_type['name'], True, self.config.WHITE)
        else:
            sel_surf = self.font.render("선택된 행성: 없음", True, self.config.WHITE)
        screen.blit(sel_surf, (10, 70))

        pygame.draw.rect(screen, self.config.WHITE, (self.config.WIDTH - self.config.SIDE_PANEL_WIDTH, 0, self.config.SIDE_PANEL_WIDTH, self.config.HEIGHT), 1)
        panel_title = self.font.render("행성 타입(1~5)", True, self.config.WHITE)
        screen.blit(panel_title, (self.config.WIDTH - self.config.SIDE_PANEL_WIDTH+10, 20))

    def render(self, screen):
        for p in self.planets:
            p.render(screen)
        self.render_ui(screen)

# ============================
# Main
# ============================
def main():
    pygame.init()
    screen = pygame.display.set_mode((Config.WIDTH, Config.HEIGHT))
    pygame.display.set_caption("SPH 행성 충돌 시뮬레이션")
    clock = pygame.time.Clock()

    gs = GameState(Config)

    # 행성 타입 (예: 태양계 행성 or 임의)
    planet_types = [
        {'name': 'Mercury', 'radius': 10, 'mass': 0.33,  'init_vel': (0.0, 0.0), 'color': (169, 169, 169)},
        {'name': 'Venus',   'radius': 20, 'mass': 4.87,  'init_vel': (0.0, 0.0), 'color': (255, 234,   0)},
        {'name': 'Earth',   'radius': 20, 'mass': 5.97,  'init_vel': (0.0, 0.0), 'color': (  0, 102, 204)},
        {'name': 'Mars',    'radius': 15, 'mass': 0.64,  'init_vel': (0.0, 0.0), 'color': (188,  39,  50)},
        {'name': 'Jupiter', 'radius': 40, 'mass': 1898,  'init_vel': (0.0, 0.0), 'color': (255,  69,   0)},
    ]

    # --- 기존에 기본 행성을 소환하던 코드를 제거하여
    #     초기에는 아무것도 소환되지 않도록 합니다.
    # 예: gs.add_planet(planet_types[1], (300,300))
    #     gs.add_planet(planet_types[2], (500,300))
    # -----

    running = True
    while running:
        dt_ms = clock.tick(Config.FPS)
        fps = clock.get_fps()
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    gs.paused = not gs.paused
                elif event.key == K_1:
                    gs.selected_planet_type = planet_types[0]
                elif event.key == K_2:
                    gs.selected_planet_type = planet_types[1]
                elif event.key == K_3:
                    gs.selected_planet_type = planet_types[2]
                elif event.key == K_4:
                    gs.selected_planet_type = planet_types[3]
                elif event.key == K_5:
                    gs.selected_planet_type = planet_types[4]

                # 예시) E 키로 폭발 모드/비폭발 모드를 토글할 수 있도록
                # 하고 싶다면 아래처럼 사용 가능
                # elif event.key == K_e:
                #     gs.explosion_mode = not gs.explosion_mode

            elif event.type == MOUSEBUTTONDOWN:
                if gs.selected_planet_type is not None:
                    mx, my = event.pos
                    # 사이드 패널 영역을 벗어나야 행성을 소환
                    if mx < Config.WIDTH - Config.SIDE_PANEL_WIDTH:
                        # 여기서 필요하다면 explosion=True로 소환
                        gs.add_planet(gs.selected_planet_type, (mx,my), explosion=False)
                        gs.selected_planet_type = None

        gs.update()

        screen.fill(Config.BG_COLOR)
        gs.render(screen)

        # FPS 표시
        fps_surf = gs.font.render(f"FPS: {fps:.1f}", True, Config.WHITE)
        screen.blit(fps_surf, (10, 100))

        # 행성 미리보기
        if gs.selected_planet_type is not None:
            mx, my = pygame.mouse.get_pos()
            if mx < Config.WIDTH - Config.SIDE_PANEL_WIDTH:
                preview_surf = pygame.Surface((Config.WIDTH, Config.HEIGHT), pygame.SRCALPHA)
                preview_surf.set_alpha(100)
                pc = gs.selected_planet_type['color']
                pygame.draw.circle(preview_surf, pc, (mx, my), gs.selected_planet_type['radius'])
                screen.blit(preview_surf, (0,0))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
