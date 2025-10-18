import jax
import jax.numpy as jnp
from . import discretize

@jax.tree_util.register_pytree_node_class
class Kf:
    def __init__(self, A, B, Q, X_initial, U_initial, P_initial, num_states: int, num_control: int, dt=None) -> None:
        self.num_states = num_states
        self.num_control = num_control
        self.X = X_initial
        self.U = U_initial
        self.P = P_initial
        self.dt = dt
        self.A, self.B = A, B if dt is None else discretize.discretizeAB(A, B, num_states, num_control, dt)
        self.Q = Q if dt is None else discretize.discretizeQ(A, Q, num_states, dt)

    def update(self, z, H, R, U, dt=None):
        if self.dt is None:
            self.A, self.B = discretize.discretizeAB(self.A, self.B, self.num_states, self.num_control, dt)
            self.Q = discretize.discretizeQ(self.A, self.Q, self.num_states, dt)

        self.U = U

        self.__predict()

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ jnp.linalg.inv(S)

        self.X = self.X + K @ (z - H @ self.X)
        self.P = (jnp.eye(K.shape[0]) - K @ H) @ self.P

    def __predict(self):
        self.X = self.A @ self.X + self.B @ self.U
        self.P = self.A @ self.P @ self.A.T + self.Q

