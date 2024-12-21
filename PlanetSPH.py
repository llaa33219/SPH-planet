import math
import random
import tkinter as tk
from tkinter import ttk

########################################
# 1) 전역 파라미터
########################################
SCREEN_WIDTH  = 1200
SCREEN_HEIGHT = 800
FPS           = 60

# 기본 SPH
KERNEL_RADIUS      = 40.0
REST_DENSITY       = 1.0
PRESSURE_STIFFNESS = 500.0
VISCOSITY          = 0.1
MASS               = 1.0
TIME_STEP          = 0.01
G                  = 0.5   # 인공 중력

# 표면장력(응집력) 계수
COHESION_COEFF     = 0.2   # 크면 파티클들이 더 강하게 달라붙음

# 행성 내부 스프링 강도
PLANET_SPRING_K    = 5.0   # 크게 할수록 행성이 잘 안 퍼짐

# 행성 옵션 (num_particles, planet_radius, color)
PLANET_OPTIONS = {
    "Earth":   (80,  40, (100,149,237)),
    "Mars":    (60,  35, (200,100,100)),
    "Jupiter": (200, 70, (220,220,150)),
    "Blue":    (60,  30, (100,100,255)),
    "Red":     (60,  30, (255,100,100)),
}

########################################
# 2) 벡터 유틸
########################################
def vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1])

def vec_add(a, b):
    return (a[0] + b[0], a[1] + b[1])

def vec_mul(a, s):
    return (a[0]*s, a[1]*s)

def vec_len(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1])

def vec_normalize(v):
    l = vec_len(v)
    if l < 1e-12:
        return (0.0, 0.0)
    return (v[0]/l, v[1]/l)

########################################
# 3) SPH 커널
########################################
def poly6_kernel(r, h):
    """기본 Poly6 커널"""
    if r < 0 or r > h:
        return 0
    k = 315.0/(64.0*math.pi*(h**9))
    return k*((h*h - r*r)**3)

########################################
# 4) 파티클 클래스
########################################
class Particle:
    """
    planet_id: 어느 '행성'에 속하는지 구분 (0이면 미분류)
    center_pos: 행성 생성 시점의 '행성 중심' 좌표
    dist_to_center: 행성 중심과의 초기 거리
    """
    def __init__(self, x, y, vx=0, vy=0, mass=MASS, color=(255,255,255),
                 planet_id=0, center_pos=(0,0), dist_to_center=0.0):
        self.pos = (x, y)
        self.vel = (vx, vy)
        self.mass = mass

        self.density = REST_DENSITY
        self.pressure = 0.0
        self.force = (0.0, 0.0)
        self.color = color

        self.planet_id = planet_id
        self.center_pos = center_pos
        self.dist_to_center = dist_to_center

    def apply_force(self):
        ax = self.force[0] / self.mass
        ay = self.force[1] / self.mass
        vx_new = self.vel[0] + ax * TIME_STEP
        vy_new = self.vel[1] + ay * TIME_STEP
        px_new = self.pos[0] + vx_new * TIME_STEP
        py_new = self.pos[1] + vy_new * TIME_STEP
        self.vel = (vx_new, vy_new)
        self.pos = (px_new, py_new)

########################################
# 5) SPH 계산 (밀도, 압력, 힘)
########################################
def compute_density_pressure(particles):
    n = len(particles)
    for p in particles:
        p.density = 0.0

    for i in range(n):
        p = particles[i]
        for j in range(n):
            if i == j:
                continue
            q = particles[j]
            rx = p.pos[0] - q.pos[0]
            ry = p.pos[1] - q.pos[1]
            r = math.hypot(rx, ry)
            if r < KERNEL_RADIUS:
                p.density += MASS * poly6_kernel(r, KERNEL_RADIUS)

        p.pressure = PRESSURE_STIFFNESS * (p.density - REST_DENSITY)

def compute_forces(particles):
    n = len(particles)
    for p in particles:
        p.force = (0.0, 0.0)

    for i in range(n):
        p = particles[i]
        fx, fy = 0.0, 0.0

        # 5.1) SPH 기본 항 (압력, 점성, 인공중력, 표면장력)
        for j in range(n):
            if i == j:
                continue
            q = particles[j]
            dx = p.pos[0] - q.pos[0]
            dy = p.pos[1] - q.pos[1]
            r = math.hypot(dx, dy)
            if 0 < r < KERNEL_RADIUS:
                dir_vec = vec_normalize((dx, dy))

                # (A) 압력(간단화)
                press_term = -0.5*(p.pressure + q.pressure)
                press_force = vec_mul(dir_vec, press_term)

                # (B) 점성(간단화)
                rel_vel = (q.vel[0]-p.vel[0], q.vel[1]-p.vel[1])
                visc_force = vec_mul(rel_vel, VISCOSITY)

                # (C) 인공 중력
                grav = G * p.mass * q.mass / (r*r)
                grav_force = vec_mul(dir_vec, grav)

                # (D) 표면장력(응집력): 가까울수록 서로 끌어당김
                cohesion_strength = COHESION_COEFF*(1.0 - (r/KERNEL_RADIUS))**2
                cohesion_force = vec_mul(dir_vec, -cohesion_strength)

                fx += (press_force[0] + visc_force[0] + grav_force[0] + cohesion_force[0])
                fy += (press_force[1] + visc_force[1] + grav_force[1] + cohesion_force[1])

        # 5.2) 행성 내부 스프링
        #      행성 생성 당시의 'center_pos'와 'dist_to_center' 정보를 사용
        if p.planet_id != 0:
            cx, cy = p.center_pos
            desired_r = p.dist_to_center  # 초기 거리
            dx = p.pos[0] - cx
            dy = p.pos[1] - cy
            current_r = math.hypot(dx, dy)
            diff = current_r - desired_r
            if abs(diff) > 1e-3:
                spring_dir = vec_normalize((dx, dy))
                spring_f = -PLANET_SPRING_K * diff  # 후크의 법칙
                fx += spring_f * spring_dir[0]
                fy += spring_f * spring_dir[1]

        p.force = (fx, fy)

########################################
# 6) 행성(=파티클 덩어리) 생성
########################################
_planet_id_counter = 1

def create_planet(planet_name, center):
    """
    planet_name: "Earth", "Mars", ...
    center: (cx, cy)
    => 여러 입자를 생성하여 하나의 '행성'으로 구성
    """
    global _planet_id_counter
    num_p, radius, color = PLANET_OPTIONS[planet_name]
    cx, cy = center
    planet_id = _planet_id_counter
    _planet_id_counter += 1

    new_particles = []
    for _ in range(num_p):
        angle = random.random()*2*math.pi
        r = radius*math.sqrt(random.random())
        x = cx + r*math.cos(angle)
        y = cy + r*math.sin(angle)
        vx = random.uniform(-1,1)*0.5
        vy = random.uniform(-1,1)*0.5

        p = Particle(
            x, y, vx, vy, MASS, color,
            planet_id=planet_id,
            center_pos=(cx, cy),
            dist_to_center=r  # 이 파티클의 '초기' 행성 중심 거리
        )
        new_particles.append(p)
    return new_particles

########################################
# 7) Tkinter GUI
########################################
class SPHPlanetSimGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SPH + Surface Tension + Planet Springs")

        self.particles = []  # 전체 파티클
        self.paused = False

        # 소환 모드
        self.spawning_planet = False
        self.planet_name = None

        # 마우스 좌표
        self.mouse_x = SCREEN_WIDTH//2
        self.mouse_y = SCREEN_HEIGHT//2

        # 상단 UI
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # 콤보박스
        self.planet_var = tk.StringVar()
        planet_list = list(PLANET_OPTIONS.keys())
        self.planet_var.set(planet_list[0])
        self.combo = ttk.Combobox(control_frame, textvariable=self.planet_var, values=planet_list, width=10)
        self.combo.pack(side=tk.LEFT, padx=5, pady=5)

        # 행성 소환 버튼
        self.spawn_btn = tk.Button(control_frame, text="행성 소환", command=self.toggle_spawning)
        self.spawn_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # 일시정지 버튼
        self.pause_btn = tk.Button(control_frame, text="일시정지", command=self.toggle_pause)
        self.pause_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # 종료 버튼
        quit_btn = tk.Button(control_frame, text="종료", command=self.root.quit)
        quit_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Canvas
        self.canvas = tk.Canvas(self.root, width=SCREEN_WIDTH, height=SCREEN_HEIGHT, bg="black")
        self.canvas.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # 마우스 이벤트
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-1>", self.on_mouse_click)

        self.update_frame()

    def toggle_spawning(self):
        self.spawning_planet = not self.spawning_planet
        if self.spawning_planet:
            self.planet_name = self.planet_var.get()
            self.spawn_btn.config(text="소환 중...(재클릭 해제)")
        else:
            self.planet_name = None
            self.spawn_btn.config(text="행성 소환")

    def on_mouse_move(self, event):
        self.mouse_x = event.x
        self.mouse_y = event.y

    def on_mouse_click(self, event):
        if self.spawning_planet and self.planet_name:
            cx, cy = (event.x, event.y)
            new_pl = create_planet(self.planet_name, (cx, cy))
            self.particles.extend(new_pl)

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.config(text=("재개" if self.paused else "일시정지"))

    def update_frame(self):
        if not self.paused:
            compute_density_pressure(self.particles)
            compute_forces(self.particles)
            for p in self.particles:
                p.apply_force()

        # 화면 그리기
        self.canvas.delete("all")

        # 모든 파티클을 작은 원으로 표현
        for p in self.particles:
            x, y = p.pos
            r = 2
            self.canvas.create_oval(x-r, y-r, x+r, y+r,
                                    fill=self.rgb_to_hex(p.color),
                                    outline="")

        # 유령 행성(테두리)
        if self.spawning_planet and self.planet_name:
            num_p, rad, col = PLANET_OPTIONS[self.planet_name]
            cx, cy = (self.mouse_x, self.mouse_y)
            self.canvas.create_oval(cx-rad, cy-rad, cx+rad, cy+rad,
                                    outline=self.rgb_to_hex(col), width=2)
            self.canvas.create_text(cx, cy-rad-10,
                                    text=f"{self.planet_name}",
                                    fill="white")

        # 안내 텍스트
        info_text = (
            "SPH + Surface Tension + Planet-Center Springs\n"
            "[행성 소환] -> Canvas 클릭 시 행성(다수 파티클) 생성\n"
            "[일시정지/재개] 시뮬레이션 토글\n"
            f"Particles: {len(self.particles)}\n"
            f"Planet Spring K = {PLANET_SPRING_K}, Cohesion = {COHESION_COEFF}\n"
        )
        self.canvas.create_text(10, 10, anchor="nw", text=info_text, fill="white")

        self.root.after(int(1000/FPS), self.update_frame)

    def rgb_to_hex(self, color):
        return "#%02x%02x%02x" % color

def main():
    root = tk.Tk()
    app = SPHPlanetSimGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
