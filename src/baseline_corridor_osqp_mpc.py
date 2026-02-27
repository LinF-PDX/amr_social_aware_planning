#!/usr/bin/env python3
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import osqp
from dataclasses import dataclass


@dataclass
class MPCConfig:
    # MPC
    dt: float = 0.1
    N: int = 20
    vref: float = 0.8  # fixed forward speed used in plant sim + linear model

    # Corridor
    W: float = 5.0  # corridor width (m)
    margin: float = 0.15 # safety margin from walls (for constraints)

    # Weights
    qy: float = 20.0    # lateral error weight
    qth: float = 2.0    # heading error weight
    rw: float = 10.0     # control effort weight
    rdw: float = 0.1    # delta control weight (smoothness)

    # Bounds
    wmax: float = 1.0   # max angular velocity (rad/s)

    # =========================
    # ### CHANGED ### APF params now "global" (not single-person)
    # =========================
    apf_sigma_x: float = 1.8        # how far along x the influence extends (m)
    apf_gain: float = 14.0          # strength of APF bias (bigger => more avoidance)
    apf_pass_offset: float = 1.0    # desired lateral clearance when passing (m)


# =========================
# ### ADDED ### Person definition + propagation
# =========================
@dataclass
class Person:
    p: np.ndarray      # current position [x, y]
    v: float           # constant speed (m/s)
    dir: np.ndarray    # direction vector [dx, dy] (will be normalized)

    def __post_init__(self):
        n = np.linalg.norm(self.dir)
        if n < 1e-9:
            raise ValueError("Person.dir must be non-zero")
        self.dir = self.dir / n


def step_person_straight(person: Person, dt: float) -> None:
    """In-place update: straight-line motion at constant speed."""
    person.p = person.p + person.v * person.dir * dt


class CorridorMPC_OSQP:
    """
    Linear MPC for straight corridor centerline tracking:
      state x = [e_y, e_theta]
      control u = [omega]
    Linearized around theta≈0, v≈vref:
      e_y(k+1)  = e_y(k) + vref*dt*e_theta(k)
      e_th(k+1) = e_th(k) + dt*omega(k)

    QP solved by OSQP. Outside the optimizer we simulate the nonlinear unicycle.
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
        self.S = np.array([[cfg.rdw]])  # penalty on delta-omega

        self._build_qp()

    def _build_qp(self):
        cfg = self.cfg
        N, nx, nu = cfg.N, self.nx, self.nu

        nX = (N + 1) * nx
        nU = N * nu
        nZ = nX + nU
        self.nX, self.nU, self.nZ = nX, nU, nZ

        # ---------------- Quadratic cost 0.5 z' P z + q' z ----------------
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

        # ---------------- Constraints l <= Acon z <= u ----------------
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

    # =========================
    # ### CHANGED ### solve() now takes a list of people (multi-agent)
    # =========================
    def solve(
        self,
        x_init: np.ndarray,
        u_prev: float,
        robot_x0: float,
        people: list
    ) -> float:
        cfg = self.cfg
        N, nx = cfg.N, self.nx

        # Initial state constraint
        self.l[self.idx_x0] = x_init
        self.u[self.idx_x0] = x_init
        self.prob.update(l=self.l, u=self.u)

        q = np.zeros(self.nZ)

        # Delta-u smoothing
        q[self.idx_u0] = -2.0 * self.S[0, 0] * u_prev

        # =========================
        # ### CHANGED ### APF: sum contributions from all people
        # =========================
        y_robot = float(x_init[0])  # e_y = y in straight corridor

        for person in people:
            x_p0, y_p0 = float(person.p[0]), float(person.p[1])
            vpx, vpy = float(person.v * person.dir[0]), float(person.v * person.dir[1])

            # choose pass side (per person) based on current relative y
            side = 1.0 if (y_robot - y_p0) >= 0.0 else -1.0
            y_des = y_p0 + side * cfg.apf_pass_offset

            for k in range(N + 1):
                x_rk = robot_x0 + cfg.vref * cfg.dt * k
                # predict person position (straight line)
                x_pk = x_p0 + vpx * cfg.dt * k
                y_pk = y_p0 + vpy * cfg.dt * k

                dx = x_rk - x_pk
                alpha = np.exp(-(dx / cfg.apf_sigma_x) ** 2)

                # Optional: also gate by lateral closeness (keeps behavior sane)
                # (still OSQP-friendly since it's just a weight computed outside the QP)
                dy0 = (y_robot - y_pk)
                beta = np.exp(-(dy0 / (cfg.apf_pass_offset + 1e-6)) ** 2)

                w = cfg.apf_gain * alpha * beta

                ix = k * nx
                q[ix + 0] += -2.0 * w * y_des

        self.prob.update(q=q)

        res = self.prob.solve()
        if res.info.status_val not in (1, 2):
            return 0.0
        return float(res.x[self.idx_u0])


def step_unicycle(state, v, w, dt):
    x, y, th = state
    x += v * np.cos(th) * dt
    y += v * np.sin(th) * dt
    th += w * dt
    th = (th + np.pi) % (2 * np.pi) - np.pi
    return np.array([x, y, th])


def main():
    cfg = MPCConfig()
    mpc = CorridorMPC_OSQP(cfg)

    # Scene
    L = 20.0
    W = cfg.W

    # Robot initial state
    state = np.array([0.0, 0.6, np.deg2rad(45.0)])
    omega_prev = 0.0

    # =========================
    # ### CHANGED ### Define ANY number of people here
    # Each person: start p0=[x,y], speed v, direction dir=[dx,dy] in global coords
    # =========================
    people = [
        Person(p=np.array([15.0, 0.2]), v=0.6, dir=np.array([-1.0, 0.0])),  # toward robot
        Person(p=np.array([3.0, -1.2]), v=0.4, dir=np.array([ 0.0, 1.0])), # crossing upward
        Person(p=np.array([18.0,  1.5]), v=0.3, dir=np.array([-1.0, -0.2])),# diagonal
    ]

    T = 18.0
    steps = int(T / cfg.dt)

    traj = np.zeros((steps + 1, 3))
    omegas = np.zeros(steps)
    traj[0] = state

    # =========================
    # ### ADDED ### store all people trajectories: list of arrays
    # =========================
    people_traj = [np.zeros((steps + 1, 2)) for _ in people]
    for i, p in enumerate(people):
        people_traj[i][0] = p.p.copy()

    # ---- live plot ----
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot([0, L], [ W/2,  W/2], linewidth=2)
    ax.plot([0, L], [-W/2, -W/2], linewidth=2)
    ax.plot([0, L], [0, 0], "--", linewidth=1.5)

    ax.set_xlim(-0.5, L+0.5)
    ax.set_ylim(-W/2-0.6, W/2+0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Corridor MPC + multiple moving people (straight-line)")

    path_line, = ax.plot([], [], linewidth=2.5)
    robot_dot, = ax.plot([], [], marker="o", markersize=8)
    heading_line, = ax.plot([], [], linewidth=2.0)

    # =========================
    # ### ADDED ### artists for multiple people
    # =========================
    person_dots = []
    person_lines = []
    person_hist = [[] for _ in people]  # list of list-of-positions for live plot
    for _ in people:
        d, = ax.plot([], [], marker="o", markersize=8)
        l, = ax.plot([], [], linewidth=1.5)
        person_dots.append(d)
        person_lines.append(l)

    xs, ys = [], []

    for k in range(steps):
        # =========================
        # ### CHANGED ### propagate all people
        # =========================
        for i, p in enumerate(people):
            step_person_straight(p, cfg.dt)
            people_traj[i][k + 1] = p.p.copy()

        x_init = np.array([state[1], state[2]])

        # =========================
        # ### CHANGED ### solve with ALL people
        # =========================
        omega = mpc.solve(
            x_init=x_init,
            u_prev=omega_prev,
            robot_x0=float(state[0]),
            people=people
        )

        state = step_unicycle(state, v=cfg.vref, w=omega, dt=cfg.dt)
        omega_prev = omega

        traj[k + 1] = state
        omegas[k] = omega

        # update robot visuals
        xs.append(state[0]); ys.append(state[1])
        path_line.set_data(xs, ys)
        robot_dot.set_data([state[0]], [state[1]])
        hx = state[0] + 0.5*np.cos(state[2])
        hy = state[1] + 0.5*np.sin(state[2])
        heading_line.set_data([state[0], hx], [state[1], hy])

        # =========================
        # ### ADDED ### update people visuals
        # =========================
        for i, p in enumerate(people):
            person_hist[i].append(p.p.copy())
            ph = np.array(person_hist[i])
            person_dots[i].set_data([p.p[0]], [p.p[1]])
            person_lines[i].set_data(ph[:, 0], ph[:, 1])

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

        if state[0] > L:
            traj = traj[:k + 2]
            omegas = omegas[:k + 1]
            for i in range(len(people_traj)):
                people_traj[i] = people_traj[i][:k + 2]
            break

    plt.ioff()

    # ---- final plot ----
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot([0, L], [W / 2, W / 2], linewidth=2, label="Wall")
    ax.plot([0, L], [-W / 2, -W / 2], linewidth=2, label="Wall")
    ax.plot([0, L], [0, 0], "--", linewidth=1.5, label="Centerline")
    ax.plot(traj[:, 0], traj[:, 1], linewidth=2.5, label="Robot path")

    # =========================
    # ### ADDED ### plot all people paths
    # =========================
    for i, ptr in enumerate(people_traj):
        ax.plot(ptr[:, 0], ptr[:, 1], linewidth=2.0, label=f"Person {i} path")

    skip = max(1, int(1.0 / (cfg.vref * cfg.dt)))
    for i in range(0, len(traj), skip):
        x, y, th = traj[i]
        ax.arrow(x, y, 0.4 * np.cos(th), 0.4 * np.sin(th),
                 head_width=0.08, length_includes_head=True)

    ax.set_xlim(-0.5, L + 0.5)
    ax.set_ylim(-W / 2 - 0.6, W / 2 + 0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Corridor MPC with multiple moving people (APF bias via q update)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # ---- time plots ----
    t = np.arange(len(traj)) * cfg.dt
    fig2, ax2 = plt.subplots(figsize=(9, 4.2))
    ax2.plot(t, traj[:, 1], label="y [m]")
    ax2.plot(t, np.rad2deg(traj[:, 2]), label="theta [deg]")
    tu = np.arange(len(omegas)) * cfg.dt
    ax2.plot(tu, np.rad2deg(omegas), label="omega [deg/s]")
    ax2.set_xlabel("time [s]")
    ax2.set_title("Tracking + control (multi-person APF bias)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.show()


if __name__ == "__main__":
    main()