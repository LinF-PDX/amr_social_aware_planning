#!/usr/bin/env python3
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import osqp
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MPCConfig:
    # MPC
    dt: float = 0.1
    N: int = 20
    vref: float = 0.8  # fixed forward speed used in plant sim + linear model

    # Corridor
    W: float = 5.0  # corridor width (m)
    margin: float = 0.15  # safety margin from walls (for constraints)

    # Weights
    qy: float = 20.0    # lateral error weight
    qth: float = 2.0    # heading error weight
    rw: float = 10.0    # control effort weight
    rdw: float = 0.1    # delta control weight (smoothness)

    # Bounds
    wmax: float = 1.0   # max angular velocity (rad/s)

    # APF (for MPC bias)
    apf_sigma_front: float = 1.2
    apf_sigma_back: float = 0.6
    apf_sigma_side: float = 0.6
    apf_gain: float = 10.0
    apf_pass_offset: float = 1.0

    # Potential field visualization
    pf_sigma_front: float = 1.2      # spread in front of the person
    pf_sigma_back: float = 0.6       # spread behind the person
    pf_sigma_side: float = 0.6       # left/right spread
    pf_grid_dx: float = 0.20         # grid resolution x
    pf_grid_dy: float = 0.10         # grid resolution y
    pf_alpha: float = 0.35           # contour alpha for individual fields
    pf_draw_every: int = 1           # draw PF every N frames (increase if slow)
    pf_show_combined: bool = True    # show summed field as filled contour

    # Estimation
    process_noise_std: tuple[float, float, float] = (0.05, 0.05, np.deg2rad(0.4))
    measurement_noise_std: tuple[float, float, float] = (0.03, 0.03, np.deg2rad(1.0))
    random_seed: int = 7

    # Person reaction to AMR
    person_reaction_front_sigma: float = 2.6
    person_reaction_back_sigma: float = 0.4
    person_reaction_side_sigma: float = 1.0
    person_reaction_turn_gain: float = 1.3
    person_reaction_slow_gain: float = 0.8
    person_reaction_min_speed: float = 0.08
    person_reaction_turn_rate: float = 5.0
    person_reaction_speed_rate: float = 4.0
    person_reaction_closing_speed: float = 0.05


@dataclass
class Person:
    p: np.ndarray      # current position [x, y]
    v: float           # constant speed (m/s)
    dir: np.ndarray    # direction [dx, dy], will be normalized
    v_nominal: float = field(init=False)
    dir_nominal: np.ndarray = field(init=False)

    def __post_init__(self):
        n = float(np.linalg.norm(self.dir))
        if n < 1e-9:
            raise ValueError("Person.dir must be non-zero")
        self.dir = self.dir / n
        self.v = float(self.v)
        self.v_nominal = self.v
        self.dir_nominal = self.dir.copy()


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def left_normal(direction: np.ndarray) -> np.ndarray:
    return np.array([-direction[1], direction[0]])


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return vector.copy()
    return vector / norm


def person_frame_coordinates(point: np.ndarray, person_pos: np.ndarray,
                             person_dir: np.ndarray) -> tuple[float, float]:
    rel = point - person_pos
    heading = person_dir
    lateral_axis = left_normal(heading)
    longitudinal = float(rel @ heading)
    lateral = float(rel @ lateral_axis)
    return longitudinal, lateral


def asymmetric_gaussian(longitudinal: np.ndarray | float, lateral: np.ndarray | float,
                        sigma_front: float, sigma_back: float,
                        sigma_side: float) -> np.ndarray | float:
    sigma_long = np.where(np.asarray(longitudinal) >= 0.0, sigma_front, sigma_back)
    return np.exp(-((np.asarray(longitudinal) / (sigma_long + 1e-9)) ** 2 +
                    (np.asarray(lateral) / (sigma_side + 1e-9)) ** 2))


class UnicycleEKF:
    """EKF for state [x, y, theta] with direct noisy measurement of the same state."""

    def __init__(self, dt: float, process_std: tuple[float, float, float],
                 measurement_std: tuple[float, float, float]):
        self.dt = dt
        self.Q = np.diag(np.square(process_std))
        self.R = np.diag(np.square(measurement_std))
        self.H = np.eye(3)
        self.state = np.zeros(3)
        self.P = np.eye(3)

    def initialize(self, measurement: np.ndarray, covariance_scale: float = 1.0) -> None:
        self.state = measurement.copy()
        self.state[2] = wrap_angle(self.state[2])
        self.P = covariance_scale * self.R.copy()

    def predict(self, v: float, w: float) -> None:
        x, y, th = self.state
        dt = self.dt

        self.state = np.array([
            x + v * np.cos(th) * dt,
            y + v * np.sin(th) * dt,
            wrap_angle(th + w * dt)
        ])

        F = np.array([
            [1.0, 0.0, -v * np.sin(th) * dt],
            [0.0, 1.0,  v * np.cos(th) * dt],
            [0.0, 0.0, 1.0]
        ])
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement: np.ndarray) -> None:
        innovation = measurement - self.H @ self.state
        innovation[2] = wrap_angle(innovation[2])

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ innovation
        self.state[2] = wrap_angle(self.state[2])

        I = np.eye(3)
        self.P = (I - K @ self.H) @ self.P


def step_person_straight(person: Person, dt: float) -> None:
    """In-place update: straight-line motion at constant speed."""
    person.p = person.p + person.v * person.dir * dt


def step_person_reactive(person: Person, robot_state: np.ndarray, robot_speed: float,
                         cfg: MPCConfig, dt: float) -> None:
    """Update a person with a simple front-aware sidestep + slowdown reaction."""
    robot_pos = robot_state[:2]
    robot_heading = np.array([np.cos(robot_state[2]), np.sin(robot_state[2])])
    robot_vel = robot_speed * robot_heading

    longitudinal, lateral = person_frame_coordinates(robot_pos, person.p, person.dir_nominal)
    rel = robot_pos - person.p
    dist = float(np.linalg.norm(rel))

    if dist > 1e-9:
        rel_hat = rel / dist
        person_vel = person.v * person.dir
        closing_speed = max(0.0, -float(rel_hat @ (robot_vel - person_vel)))
    else:
        closing_speed = robot_speed

    threat = float(asymmetric_gaussian(
        longitudinal=longitudinal,
        lateral=lateral,
        sigma_front=cfg.person_reaction_front_sigma,
        sigma_back=cfg.person_reaction_back_sigma,
        sigma_side=cfg.person_reaction_side_sigma
    ))
    if closing_speed <= cfg.person_reaction_closing_speed:
        threat = 0.0

    nominal_dir = person.dir_nominal
    nominal_left = left_normal(nominal_dir)
    side_away = -1.0 if lateral >= 0.0 else 1.0

    desired_dir = normalize_vector(
        nominal_dir + cfg.person_reaction_turn_gain * threat * side_away * nominal_left
    )
    desired_speed = max(
        cfg.person_reaction_min_speed,
        person.v_nominal * (1.0 - cfg.person_reaction_slow_gain * threat)
    )

    turn_blend = np.clip(cfg.person_reaction_turn_rate * dt, 0.0, 1.0)
    speed_blend = np.clip(cfg.person_reaction_speed_rate * dt, 0.0, 1.0)
    person.dir = normalize_vector((1.0 - turn_blend) * person.dir + turn_blend * desired_dir)
    person.v = (1.0 - speed_blend) * person.v + speed_blend * desired_speed
    person.p = person.p + person.v * person.dir * dt


class CorridorMPC_OSQP:
    """
    Linear MPC for straight corridor centerline tracking:
      state x = [e_y, e_theta]
      control u = [omega]
    Linearized around theta≈0, v≈vref:
      e_y(k+1)  = e_y(k) + vref*dt*e_theta(k)
      e_th(k+1) = e_th(k) + dt*omega(k)
    """

    def __init__(self, cfg: MPCConfig):
        self.cfg = cfg
        self.nx, self.nu = 2, 1

        dt, vref = cfg.dt, cfg.vref
        self.A = np.array([[1.0, vref * dt],
                           [0.0, 1.0]])
        self.B = np.array([[0.0],
                           [dt]])

        self.Q = np.diag([cfg.qy, cfg.qth])
        self.R = np.array([[cfg.rw]])
        self.S = np.array([[cfg.rdw]])

        self._build_qp()

    def _build_qp(self):
        cfg = self.cfg
        N, nx, nu = cfg.N, self.nx, self.nu

        nX = (N + 1) * nx
        nU = N * nu
        nZ = nX + nU
        self.nX, self.nU, self.nZ = nX, nU, nZ

        # Cost: 0.5 z' P z + q' z
        P = sp.lil_matrix((nZ, nZ))

        # State cost
        for k in range(N + 1):
            ix = k * nx
            P[ix:ix + nx, ix:ix + nx] += self.Q

        # Control cost
        for k in range(N):
            iu = nX + k * nu
            P[iu:iu + nu, iu:iu + nu] += self.R

        # Delta-u cost
        for k in range(N):
            iu = nX + k * nu
            P[iu:iu + nu, iu:iu + nu] += self.S
            if k > 0:
                iu_prev = nX + (k - 1) * nu
                P[iu:iu + nu, iu_prev:iu_prev + nu] += -self.S
                P[iu_prev:iu_prev + nu, iu:iu + nu] += -self.S
                P[iu_prev:iu_prev + nu, iu_prev:iu_prev + nu] += self.S

        self.P = P.tocsc()
        self.q = np.zeros(nZ)

        # Constraints l <= A z <= u
        constr = []
        l = []
        u = []

        # (0) Initial state equality x0 == x_init
        A_x0 = sp.lil_matrix((nx, nZ))
        A_x0[:, 0:nx] = sp.eye(nx)
        constr.append(A_x0)
        l += [0.0] * nx
        u += [0.0] * nx

        # (1) Dynamics
        for k in range(N):
            row = sp.lil_matrix((nx, nZ))
            ixk = k * nx
            ixk1 = (k + 1) * nx
            iuk = nX + k * nu

            row[:, ixk1:ixk1 + nx] = sp.eye(nx)
            row[:, ixk:ixk + nx] = -self.A
            row[:, iuk:iuk + nu] = -self.B

            constr.append(row)
            l += [0.0] * nx
            u += [0.0] * nx

        # (2) Input bounds
        for k in range(N):
            row = sp.lil_matrix((nu, nZ))
            iuk = nX + k * nu
            row[:, iuk:iuk + nu] = sp.eye(nu)
            constr.append(row)
            l += [-cfg.wmax]
            u += [cfg.wmax]

        # (3) Corridor bounds on e_y
        eymax = cfg.W / 2.0 - cfg.margin
        for k in range(N + 1):
            row = sp.lil_matrix((1, nZ))
            ix = k * nx
            row[0, ix + 0] = 1.0
            constr.append(row)
            l += [-eymax]
            u += [eymax]

        self.Acon = sp.vstack(constr).tocsc()
        self.l = np.array(l, dtype=float)
        self.u = np.array(u, dtype=float)

        self.idx_x0 = slice(0, nx)
        self.idx_u0 = nX

        self.prob = osqp.OSQP()
        self.prob.setup(
            P=self.P,
            q=self.q,
            A=self.Acon,
            l=self.l,
            u=self.u,
            verbose=False,
            warm_start=True,
            polish=True
        )

    def solve(self, x_init: np.ndarray, u_prev: float, robot_x0: float, people: list[Person]) -> float:
        cfg = self.cfg
        N, nx = cfg.N, self.nx

        # Update initial state equality constraints
        self.l[self.idx_x0] = x_init
        self.u[self.idx_x0] = x_init
        self.prob.update(l=self.l, u=self.u)

        q = np.zeros(self.nZ)

        # Delta-u smoothing for k=0
        q[self.idx_u0] = -2.0 * self.S[0, 0] * u_prev

        # Multi-person APF bias on e_y trajectory (QP stays convex)
        y_robot = float(x_init[0])  # e_y = y in straight corridor

        for person in people:
            p0 = person.p.copy()
            heading = person.dir.copy()
            lateral_axis = left_normal(heading)
            vpx, vpy = float(person.v * person.dir[0]), float(person.v * person.dir[1])

            robot_now = np.array([robot_x0, y_robot])
            _, lateral_now = person_frame_coordinates(robot_now, p0, heading)

            # keep the pass preference left/right symmetric in the person's local frame
            side = 1.0 if lateral_now >= 0.0 else -1.0

            for k in range(N + 1):
                robot_nominal = np.array([robot_x0 + cfg.vref * cfg.dt * k, y_robot])
                person_pred = p0 + np.array([vpx, vpy]) * cfg.dt * k

                longitudinal, lateral = person_frame_coordinates(
                    robot_nominal, person_pred, heading
                )
                w = cfg.apf_gain * asymmetric_gaussian(
                    longitudinal=longitudinal,
                    lateral=lateral,
                    sigma_front=cfg.apf_sigma_front,
                    sigma_back=cfg.apf_sigma_back,
                    sigma_side=cfg.apf_sigma_side
                )

                desired_pos = person_pred + longitudinal * heading + side * cfg.apf_pass_offset * lateral_axis
                y_des = float(desired_pos[1])

                ix = k * nx
                q[ix + 0] += -2.0 * w * y_des

        self.prob.update(q=q)

        res = self.prob.solve()
        if res.info.status_val not in (1, 2):
            return 0.0
        return float(res.x[self.idx_u0])


def step_unicycle(state, v, w, dt):
    """state = [x, y, theta]"""
    x, y, th = state
    x += v * np.cos(th) * dt
    y += v * np.sin(th) * dt
    th += w * dt
    th = wrap_angle(th)
    return np.array([x, y, th])


def person_potential_on_grid(X, Y, person_xy, person_dir, sigma_front, sigma_back, sigma_side):
    rel_x = X - person_xy[0]
    rel_y = Y - person_xy[1]
    heading = person_dir
    lateral_axis = left_normal(heading)

    longitudinal = rel_x * heading[0] + rel_y * heading[1]
    lateral = rel_x * lateral_axis[0] + rel_y * lateral_axis[1]
    return asymmetric_gaussian(longitudinal, lateral, sigma_front, sigma_back, sigma_side)


def _remove_contour_objects(objs):
    """
    Robust removal across matplotlib versions:
    - prefer obj.remove()
    - fallback to removing contained artists if present.
    """
    for obj in objs:
        try:
            obj.remove()
            continue
        except Exception:
            pass

        for attr in ("collections", "artists"):
            if hasattr(obj, attr):
                for a in getattr(obj, attr):
                    try:
                        a.remove()
                    except Exception:
                        pass


def main():
    cfg = MPCConfig()
    mpc = CorridorMPC_OSQP(cfg)
    ekf = UnicycleEKF(
        dt=cfg.dt,
        process_std=cfg.process_noise_std,
        measurement_std=cfg.measurement_noise_std
    )
    rng = np.random.default_rng(cfg.random_seed)
    is_headless = "agg" in plt.get_backend().lower()

    # Scene
    L = 20.0
    W = cfg.W

    # Initial robot state in world
    true_state = np.array([0.0, 0.6, np.deg2rad(45.0)])  # x, y, theta
    measured_state = true_state.copy()
    estimated_state = measured_state.copy()
    ekf.initialize(measured_state)
    omega_prev = 0.0

    # Define ANY number of people here
    # Each person: start p=[x,y], speed v, direction dir=[dx,dy] in global coords
    people: list[Person] = [
        #Person(p=np.array([15.0, 0.2]), v=0.6, dir=np.array([-1.0, 0.0])),   # toward robot
        #Person(p=np.array([12.0, -1.0]), v=0.45, dir=np.array([-1.0, 0.15])), # diagonal toward corridor
        #Person(p=np.array([10.0,  1.3]), v=0.35, dir=np.array([ 0.0, -1.0])), # crossing down
        Person(p=np.array([15.0, 0.2]), v=0.6, dir=np.array([-1.0, 0.0])),  # toward robot
        Person(p=np.array([3.0, -1.2]), v=0.4, dir=np.array([ 0.0, 1.0])), # crossing upward
        Person(p=np.array([18.0,  1.5]), v=0.3, dir=np.array([-1.0, -0.2])),# diagonal
    ]

    T = 20.0
    steps = int(T / cfg.dt)

    true_traj = np.zeros((steps + 1, 3))
    measured_traj = np.zeros((steps + 1, 3))
    estimated_traj = np.zeros((steps + 1, 3))
    omegas = np.zeros(steps)
    true_traj[0] = true_state
    measured_traj[0] = measured_state
    estimated_traj[0] = estimated_state

    # store all people trajectories
    people_traj = [np.zeros((steps + 1, 2)) for _ in people]
    for i, p in enumerate(people):
        people_traj[i][0] = p.p.copy()

    # ---- live plot ----
    if not is_headless:
        plt.ion()
    fig, ax = plt.subplots(figsize=(9, 4.2))

    # Static corridor drawing
    ax.plot([0, L], [ W/2,  W/2], linewidth=2)
    ax.plot([0, L], [-W/2, -W/2], linewidth=2)
    ax.plot([0, L], [0, 0], "--", linewidth=1.5)

    ax.set_xlim(-0.5, L+0.5)
    ax.set_ylim(-W/2-0.6, W/2+0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Corridor MPC + multiple people + potential fields")

    # Live artists
    true_path_line, = ax.plot([], [], linewidth=2.0, label="True AMR path")
    est_path_line, = ax.plot([], [], "--", linewidth=2.0, label="Estimated AMR path")
    robot_dot, = ax.plot([], [], marker="o", markersize=8, label="Estimated AMR")
    heading_line, = ax.plot([], [], linewidth=2.0)

    # People artists
    person_dots = []
    person_lines = []
    person_hist = [[] for _ in people]
    for _ in people:
        d, = ax.plot([], [], marker="o", markersize=8)
        l, = ax.plot([], [], linewidth=1.5)
        person_dots.append(d)
        person_lines.append(l)

    true_xs, true_ys = [], []
    est_xs, est_ys = [], []

    # Precompute PF grid (once)
    xg = np.arange(-0.5, L + 0.5 + cfg.pf_grid_dx, cfg.pf_grid_dx)
    yg = np.arange(-W/2 - 0.6, W/2 + 0.6 + cfg.pf_grid_dy, cfg.pf_grid_dy)
    X, Y = np.meshgrid(xg, yg)

    # We'll store contour sets here so we can delete/redraw safely
    pf_objects = []

    # contour levels
    levels_individual = np.array([0.2, 0.4, 0.6, 0.8])
    levels_combined = np.array([0.3, 0.6, 1.0, 1.6, 2.4])

    measurement_noise = np.array(cfg.measurement_noise_std)

    for k in range(steps):
        # propagate all people
        for i, p in enumerate(people):
            step_person_reactive(p, true_state, cfg.vref, cfg, cfg.dt)
            people_traj[i][k + 1] = p.p.copy()

        # MPC solve
        x_init = np.array([estimated_state[1], estimated_state[2]])
        omega = mpc.solve(
            x_init=x_init,
            u_prev=omega_prev,
            robot_x0=float(estimated_state[0]),
            people=people
        )

        # Simulate true robot motion, then create a noisy measurement.
        true_state = step_unicycle(true_state, v=cfg.vref, w=omega, dt=cfg.dt)
        noise = rng.normal(size=3) * measurement_noise
        measured_state = true_state + noise
        measured_state[2] = wrap_angle(measured_state[2])

        ekf.predict(v=cfg.vref, w=omega)
        ekf.update(measured_state)
        estimated_state = ekf.state.copy()
        omega_prev = omega

        true_traj[k + 1] = true_state
        measured_traj[k + 1] = measured_state
        estimated_traj[k + 1] = estimated_state
        omegas[k] = omega

        # Update robot visuals
        true_xs.append(true_state[0]); true_ys.append(true_state[1])
        est_xs.append(estimated_state[0]); est_ys.append(estimated_state[1])
        true_path_line.set_data(true_xs, true_ys)
        est_path_line.set_data(est_xs, est_ys)
        robot_dot.set_data([estimated_state[0]], [estimated_state[1]])
        hx = estimated_state[0] + 0.5 * np.cos(estimated_state[2])
        hy = estimated_state[1] + 0.5 * np.sin(estimated_state[2])
        heading_line.set_data([estimated_state[0], hx], [estimated_state[1], hy])

        # Update people visuals
        for i, p in enumerate(people):
            person_hist[i].append(p.p.copy())
            ph = np.array(person_hist[i])
            person_dots[i].set_data([p.p[0]], [p.p[1]])
            person_lines[i].set_data(ph[:, 0], ph[:, 1])

        # Redraw potential fields
        if (k % cfg.pf_draw_every) == 0:
            _remove_contour_objects(pf_objects)
            pf_objects = []

            U_sum = np.zeros_like(X, dtype=float)

            # per-person contour lines
            for p in people:
                Ui = person_potential_on_grid(
                    X, Y, p.p, p.dir,
                    cfg.pf_sigma_front, cfg.pf_sigma_back, cfg.pf_sigma_side
                )
                U_sum += Ui
                cs = ax.contour(X, Y, Ui, levels=levels_individual, alpha=cfg.pf_alpha)
                pf_objects.append(cs)

            # optional combined filled contour
            if cfg.pf_show_combined:
                cf = ax.contourf(X, Y, U_sum, levels=levels_combined, alpha=0.18)
                pf_objects.append(cf)

        if not is_headless:
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

        if true_state[0] > L:
            true_traj = true_traj[:k + 2]
            measured_traj = measured_traj[:k + 2]
            estimated_traj = estimated_traj[:k + 2]
            omegas = omegas[:k + 1]
            for i in range(len(people_traj)):
                people_traj[i] = people_traj[i][:k + 2]
            break

    if not is_headless:
        plt.ioff()

    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(exist_ok=True)

    # ---- final plot ----
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot([0, L], [W / 2, W / 2], linewidth=2, label="Wall")
    ax.plot([0, L], [-W / 2, -W / 2], linewidth=2, label="Wall")
    ax.plot([0, L], [0, 0], "--", linewidth=1.5, label="Centerline")
    ax.plot(true_traj[:, 0], true_traj[:, 1], linewidth=2.5, label="True AMR path")
    ax.plot(estimated_traj[:, 0], estimated_traj[:, 1], "--", linewidth=2.5, label="EKF estimate")
    # ax.plot(measured_traj[:, 0], measured_traj[:, 1], ":", linewidth=1.6, alpha=0.75, label="Noisy measurement")

    for i, ptr in enumerate(people_traj):
        ax.plot(ptr[:, 0], ptr[:, 1], linewidth=2.0, label=f"Person {i} path")

    # Heading arrows every ~1m
    skip = max(1, int(1.0 / (cfg.vref * cfg.dt)))
    for i in range(0, len(estimated_traj), skip):
        x, y, th = estimated_traj[i]
        ax.arrow(x, y, 0.4 * np.cos(th), 0.4 * np.sin(th),
                 head_width=0.08, length_includes_head=True)

    ax.set_xlim(-0.5, L + 0.5)
    ax.set_ylim(-W / 2 - 0.6, W / 2 + 0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("AMR trajectory: true vs noisy measurement vs EKF estimate")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    traj_plot_path = output_dir / "amr_true_vs_estimated_xy.png"
    fig.savefig(traj_plot_path, dpi=180, bbox_inches="tight")

    # ---- time plot ----
    t = np.arange(len(true_traj)) * cfg.dt
    fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(9, 7.0), sharex=True)
    ax2.plot(t, true_traj[:, 0], linewidth=2.0, label="True x")
    ax2.plot(t, estimated_traj[:, 0], "--", linewidth=2.0, label="Estimated x")
    ax2.plot(t, true_traj[:, 1], linewidth=2.0, label="True y")
    ax2.plot(t, estimated_traj[:, 1], "--", linewidth=2.0, label="Estimated y")
    ax2.set_ylabel("position [m]")
    ax2.set_title("Position estimate tracking")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    ax3.plot(t, np.rad2deg(true_traj[:, 2]), linewidth=2.0, label="True theta")
    ax3.plot(t, np.rad2deg(estimated_traj[:, 2]), "--", linewidth=2.0, label="Estimated theta")
    tu = np.arange(len(omegas)) * cfg.dt
    ax3.plot(tu, np.rad2deg(omegas), linewidth=1.4, label="omega [deg/s]")
    ax3.set_xlabel("time [s]")
    ax3.set_ylabel("angle [deg]")
    ax3.set_title("Heading estimate and control")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best")
    fig2.tight_layout()
    time_plot_path = output_dir / "amr_true_vs_estimated_time.png"
    fig2.savefig(time_plot_path, dpi=180, bbox_inches="tight")

    if not is_headless:
        plt.show()


if __name__ == "__main__":
    main()
