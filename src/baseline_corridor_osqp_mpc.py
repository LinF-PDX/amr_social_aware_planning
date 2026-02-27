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
    rw: float = 0.2     # control effort weight
    rdw: float = 1.1    # delta control weight (smoothness)

    # Bounds
    wmax: float = 1.0   # max angular velocity (rad/s)

    # =========================
    # ### ADDED ### APF / person simulation parameters
    # =========================
    v_person: float = -0.6          # person moving toward robot (negative x direction)
    apf_sigma_x: float = 1.8        # how far along x the influence extends (m)
    apf_gain: float = 14.0          # strength of APF bias (bigger => more avoidance)
    apf_pass_offset: float = 1.0    # desired lateral clearance when passing (m)
    person_y_amp: float = 0.25      # person oscillates near centerline
    person_y_freq: float = 0.35     # rad/s


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
        # decision z = [x0..xN, u0..u_{N-1}]
        P = sp.lil_matrix((nZ, nZ))

        # State cost
        for k in range(N + 1):
            ix = k * nx
            P[ix:ix + nx, ix:ix + nx] += self.Q

        # Control cost
        for k in range(N):
            iu = nX + k * nu
            P[iu:iu + nu, iu:iu + nu] += self.R

        # Delta-u cost: sum (u_k - u_{k-1})' S (u_k - u_{k-1})
        # For k=0, u_{-1}=u_prev is handled via q update each solve.
        for k in range(N):
            iu = nX + k * nu
            P[iu:iu + nu, iu:iu + nu] += self.S
            if k > 0:
                iu_prev = nX + (k - 1) * nu
                P[iu:iu + nu, iu_prev:iu_prev + nu] += -self.S
                P[iu_prev:iu_prev + nu, iu:iu + nu] += -self.S
                P[iu_prev:iu_prev + nu, iu_prev:iu_prev + nu] += self.S

        self.P = P.tocsc()

        # q will be updated online
        self.q = np.zeros(nZ)

        # ---------------- Constraints l <= Acon z <= u ----------------
        constr = []
        l = []
        u = []

        # (0) Initial state equality x0 == x_init (nx rows)
        A_x0 = sp.lil_matrix((nx, nZ))
        A_x0[:, 0:nx] = sp.eye(nx)
        constr.append(A_x0)
        l += [0.0] * nx  # will overwrite each solve
        u += [0.0] * nx

        # (1) Dynamics: x_{k+1} - A x_k - B u_k = 0, for k=0..N-1
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

        # (2) Input bounds: -wmax <= u_k <= wmax
        for k in range(N):
            row = sp.lil_matrix((nu, nZ))
            iuk = nX + k * nu
            row[:, iuk:iuk + nu] = sp.eye(nu)
            constr.append(row)
            l += [-cfg.wmax]
            u += [cfg.wmax]

        # (3) Corridor bounds on e_y: |e_y| <= (W/2 - margin)
        eymax = cfg.W / 2.0 - cfg.margin
        for k in range(N + 1):
            row = sp.lil_matrix((1, nZ))
            ix = k * nx
            row[0, ix + 0] = 1.0  # e_y
            constr.append(row)
            l += [-eymax]
            u += [eymax]

        self.Acon = sp.vstack(constr).tocsc()
        self.l = np.array(l, dtype=float)
        self.u = np.array(u, dtype=float)

        # for fast updates
        self.idx_x0 = slice(0, nx)     # first nx constraints are x0 equalities
        self.idx_u0 = nX               # first control index in z

        # Setup OSQP
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
    # ### CHANGED ### solve() signature to accept world x0 and person state
    # =========================
    def solve(
        self,
        x_init: np.ndarray,
        u_prev: float,
        robot_x0: float,
        person_xy: np.ndarray,
        person_vx: float
    ) -> float:
        """
        x_init = [e_y, e_theta]
        u_prev = previous applied omega
        robot_x0 = robot world x at current time (used for predicted x_k)
        person_xy = [x_p, y_p] at current time
        person_vx = constant person velocity along x (toward robot)
        returns omega_0
        """
        cfg = self.cfg
        N, nx = cfg.N, self.nx

        # Update initial-state equality constraints
        self.l[self.idx_x0] = x_init
        self.u[self.idx_x0] = x_init
        self.prob.update(l=self.l, u=self.u)

        # -------------------------
        # ### CHANGED ### q update: keep delta-u term AND add APF bias on e_y trajectory
        # -------------------------
        q = np.zeros(self.nZ)

        # (A) Delta-u smoothing term for k=0
        q[self.idx_u0] = -2.0 * self.S[0, 0] * u_prev

        # (B) APF-like "social avoidance" as a linear bias on e_y(k)
        # We keep the QP structure (same P) and inject a time-varying preference:
        #   encourage e_y(k) ≈ y_desired(k) when predicted x is close to the person.
        #
        # Implementation:
        #   Add term  apf_gain*alpha_k * (e_y(k) - y_des)^2
        # But we *approximate* it in OSQP-friendly form by only injecting the linear part
        # (the quadratic curvature already exists via Qy*e_y^2).
        #
        # Equivalent effect: shift the optimum laterally when alpha_k is high.
        x_p0, y_p0 = float(person_xy[0]), float(person_xy[1])

        # choose pass side based on current relative y (stable choice, avoids dithering)
        side = 1.0 if (x_init[0] - y_p0) >= 0.0 else -1.0  # uses e_y = y
        # desired lateral offset to pass the person
        y_des_base = y_p0 + side * cfg.apf_pass_offset

        for k in range(N + 1):
            # predicted robot and person x positions at step k
            x_rk = robot_x0 + cfg.vref * cfg.dt * k
            x_pk = x_p0 + person_vx * cfg.dt * k

            dx = x_rk - x_pk

            # influence along x (Gaussian window)
            alpha = np.exp(-(dx / cfg.apf_sigma_x) ** 2)

            # apply as a linear bias on e_y(k):
            # cost contribution ~ -2 * apf_gain*alpha * y_des * e_y
            # (so the optimizer is "pulled" toward y_des when alpha is high)
            ix = k * nx
            q[ix + 0] += -2.0 * cfg.apf_gain * alpha * y_des_base

        self.prob.update(q=q)

        res = self.prob.solve()
        if res.info.status_val not in (1, 2):  # solved / solved inaccurate
            return 0.0

        return float(res.x[self.idx_u0])


def step_unicycle(state, v, w, dt):
    """state = [x, y, theta]"""
    x, y, th = state
    x += v * np.cos(th) * dt
    y += v * np.sin(th) * dt
    th += w * dt
    th = (th + np.pi) % (2 * np.pi) - np.pi
    return np.array([x, y, th])


# =========================
# ### ADDED ### person motion model (simple)
# =========================
def step_person(person_xy, vx, t, cfg: MPCConfig, dt):
    """
    person_xy = [x_p, y_p]
    Person walks along -x (toward robot) with small y oscillation near centerline.
    """
    x_p, _ = person_xy
    x_p = x_p + vx * dt
    y_p = cfg.person_y_amp * np.sin(cfg.person_y_freq * t)
    return np.array([x_p, y_p])


def main():
    cfg = MPCConfig()
    mpc = CorridorMPC_OSQP(cfg)

    # Scene
    L = 20.0
    W = cfg.W

    # Initial robot state in world
    state = np.array([0.0, 0.6, np.deg2rad(45.0)])  # x, y, theta
    omega_prev = 0.0

    # =========================
    # ### ADDED ### initialize a person ahead, near centerline, moving toward robot
    # =========================
    person = np.array([15.0, 0.0])    # x_p, y_p
    person_vx = cfg.v_person

    T = 18.0
    steps = int(T / cfg.dt)

    traj = np.zeros((steps + 1, 3))
    omegas = np.zeros(steps)
    traj[0] = state

    # =========================
    # ### ADDED ### store person trajectory for final plot
    # =========================
    person_traj = np.zeros((steps + 1, 2))
    person_traj[0] = person

    # ---- 2D scene plot ----
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 4.2))

    # Static corridor drawing (only once)
    ax.plot([0, L], [ W/2,  W/2], linewidth=2)
    ax.plot([0, L], [-W/2, -W/2], linewidth=2)
    ax.plot([0, L], [0, 0], "--", linewidth=1.5)

    ax.set_xlim(-0.5, L+0.5)
    ax.set_ylim(-W/2-0.6, W/2+0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Real-time corridor MPC + moving person (APF bias)")

    # Live artists
    path_line, = ax.plot([], [], linewidth=2.5)             # robot trail
    robot_dot, = ax.plot([], [], marker="o", markersize=8)  # robot position
    heading_line, = ax.plot([], [], linewidth=2.0)          # robot heading

    # =========================
    # ### ADDED ### person artists
    # =========================
    person_dot, = ax.plot([], [], marker="o", markersize=8)  # person position
    person_line, = ax.plot([], [], linewidth=1.5)            # person trail

    xs, ys = [], []
    pxs, pys = [], []

    for k in range(steps):
        t_now = k * cfg.dt

        # =========================
        # ### ADDED ### update person motion first (so MPC sees current person)
        # =========================
        person = step_person(person, person_vx, t_now, cfg, cfg.dt)
        person_traj[k + 1] = person

        # Errors relative to straight corridor centerline: e_y = y, e_theta = theta
        x_init = np.array([state[1], state[2]])

        # =========================
        # ### CHANGED ### call solve() with robot_x0 + person info
        # =========================
        omega = mpc.solve(
            x_init=x_init,
            u_prev=omega_prev,
            robot_x0=float(state[0]),
            person_xy=person,
            person_vx=person_vx
        )

        state = step_unicycle(state, v=cfg.vref, w=omega, dt=cfg.dt)
        omega_prev = omega

        traj[k + 1] = state
        omegas[k] = omega

        # Update live plot
        xs.append(state[0]); ys.append(state[1])
        path_line.set_data(xs, ys)
        robot_dot.set_data([state[0]], [state[1]])
        hx = state[0] + 0.5*np.cos(state[2])
        hy = state[1] + 0.5*np.sin(state[2])
        heading_line.set_data([state[0], hx], [state[1], hy])

        # =========================
        # ### ADDED ### update person visuals
        # =========================
        pxs.append(person[0]); pys.append(person[1])
        person_dot.set_data([person[0]], [person[1]])
        person_line.set_data(pxs, pys)

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

        if state[0] > L:
            traj = traj[:k + 2]
            omegas = omegas[:k + 1]
            person_traj = person_traj[:k + 2]
            break

    plt.ioff()

    # Final trajectory plot
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot([0, L], [W / 2, W / 2], linewidth=2, label="Wall")
    ax.plot([0, L], [-W / 2, -W / 2], linewidth=2, label="Wall")
    ax.plot([0, L], [0, 0], "--", linewidth=1.5, label="Centerline")
    ax.plot(traj[:, 0], traj[:, 1], linewidth=2.5, label="Robot path")

    # =========================
    # ### ADDED ### person path
    # =========================
    ax.plot(person_traj[:, 0], person_traj[:, 1], linewidth=2.0, label="Person path")

    # Heading arrows every ~1m
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
    ax.set_title("Corridor MPC with moving person (APF bias via q update)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # ---- Time plots ----
    t = np.arange(len(traj)) * cfg.dt
    fig2, ax2 = plt.subplots(figsize=(9, 4.2))
    ax2.plot(t, traj[:, 1], label="y [m]")
    ax2.plot(t, np.rad2deg(traj[:, 2]), label="theta [deg]")
    tu = np.arange(len(omegas)) * cfg.dt
    ax2.plot(tu, np.rad2deg(omegas), label="omega [deg/s]")
    ax2.set_xlabel("time [s]")
    ax2.set_title("Tracking + control (with APF bias)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.show()


if __name__ == "__main__":
    main()