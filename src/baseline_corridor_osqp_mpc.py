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
    W: float = 2.0
    margin: float = 0.15

    # Weights
    qy: float = 20.0
    qth: float = 2.0
    rw: float = 0.2
    rdw: float = 0.1

    # Bounds
    wmax: float = 1.0


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

        # q will be updated online to incorporate u_prev in (u0-u_prev)^2
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

    def solve(self, x_init: np.ndarray, u_prev: float) -> float:
        """
        x_init = [e_y, e_theta]
        u_prev = previous applied omega
        returns omega_0
        """
        # Update initial-state equality constraints
        self.l[self.idx_x0] = x_init
        self.u[self.idx_x0] = x_init
        self.prob.update(l=self.l, u=self.u)

        # Update q for (u0 - u_prev)' S (u0 - u_prev)
        # expands: u0' S u0 - 2 u_prev' S u0 + const
        q = np.zeros(self.nZ)
        q[self.idx_u0] = -2.0 * self.S[0, 0] * u_prev
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


def main():
    cfg = MPCConfig()
    mpc = CorridorMPC_OSQP(cfg)

    # Scene
    L = 20.0
    W = cfg.W

    # Initial robot state in world
    state = np.array([0.0, 0.6, np.deg2rad(30.0)])  # x, y, theta
    omega_prev = 0.0

    T = 18.0
    steps = int(T / cfg.dt)

    traj = np.zeros((steps + 1, 3))
    omegas = np.zeros(steps)
    traj[0] = state

    # ---- 2D scene plot ----
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 4.2))

    L = 20.0
    W = cfg.W

    # Static corridor drawing (only once)
    ax.plot([0, L], [ W/2,  W/2], linewidth=2)
    ax.plot([0, L], [-W/2, -W/2], linewidth=2)
    ax.plot([0, L], [0, 0], "--", linewidth=1.5)

    ax.set_xlim(-0.5, L+0.5)
    ax.set_ylim(-W/2-0.6, W/2+0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Real-time corridor MPC")

    # Live artists (updated each step)
    path_line, = ax.plot([], [], linewidth=2.5)           # robot trail
    robot_dot, = ax.plot([], [], marker="o", markersize=8) # robot position
    heading_line, = ax.plot([], [], linewidth=2.0)         # heading arrow (as a line)

    xs, ys = [], []

    for k in range(steps):
        # Errors relative to straight corridor centerline: e_y = y, e_theta = theta
        x_init = np.array([state[1], state[2]])
        omega = mpc.solve(x_init, omega_prev)

        state = step_unicycle(state, v=cfg.vref, w=omega, dt=cfg.dt)
        omega_prev = omega

        traj[k + 1] = state
        omegas[k] = omega

        # Update live plot
        xs.append(state[0]); ys.append(state[1])
        path_line.set_data(xs, ys)
        robot_dot.set_data([state[0]], [state[1]])
        # heading indicator
        hx = state[0] + 0.5*np.cos(state[2])
        hy = state[1] + 0.5*np.sin(state[2])
        heading_line.set_data([state[0], hx], [state[1], hy])

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)   # controls refresh rate

        if state[0] > L:
            traj = traj[:k + 2]
            omegas = omegas[:k + 1]
            break
    
    plt.ioff()

    # Final trajectory plot
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot([0, L], [W / 2, W / 2], linewidth=2, label="Wall")
    ax.plot([0, L], [-W / 2, -W / 2], linewidth=2, label="Wall")
    ax.plot([0, L], [0, 0], "--", linewidth=1.5, label="Centerline")
    ax.plot(traj[:, 0], traj[:, 1], linewidth=2.5, label="Robot path")

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
    ax.set_title("Baseline corridor tracking (OSQP linear MPC, nonlinear unicycle sim)")
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
    ax2.set_title("Tracking errors and control")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.show()


if __name__ == "__main__":
    main()
