import jax.numpy as jnp
import jax
from . import discretize
from functools import partial

@jax.tree_util.register_pytree_node_class
class Ekf:
    def __init__(self, F_c, Q_c, X_initial, U_initial, P_initial, num_states: int, num_control: int, initial_dt=0.1) -> None:
       self.X = X_initial
       self.P = P_initial
       self.U = U_initial
       self.F_c = F_c
       self.Q_c = Q_c
       self.num_states = num_states
       self.num_control = num_control
       self.J = jax.jacrev(self.F_c, argnums = 0)
       self.X_predicted = X_initial
       self.P_predicted = P_initial
       self.last_dt = initial_dt

    def tree_flatten(self):
        return ((self.X, self.U, self.P, self.X_predicted, self.P_predicted, self.last_dt), (self.F_c, self.Q_c, self.J, self.num_states, self.num_control))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        ekf = object.__new__(Ekf)
        ekf.X, ekf.U, ekf.P, ekf.X_predicted, ekf.P_predicted, ekf.last_dt = children
        ekf.F_c, ekf.Q_c, ekf.J, ekf.num_states, ekf.num_control = aux_data
        return ekf

def __update(ekf, U, dt):
        ekf.U = U
        ekf = __predict(ekf, dt)
        ekf.last_dt = dt
        return ekf

def __nothing(ekf, U, dt):
    return ekf

@partial(jax.jit, static_argnames=['stable'])
def update_linear(ekf,z, H, R, U, dt, stable=False):
        # time update
        ekf = jax.lax.cond(dt > 0, __update, __nothing, ekf, U, dt)

        R_d = discretize.discretizeR(R, ekf.last_dt)

        # measurement update
        R_e = R_d + H @ ekf.P @ H.T
        K = ekf.P @ H.T @ jnp.linalg.inv(R_e)
        e = z - H @ ekf.X # innovations fo the KF

        ekf.X = ekf.X + K @ e
        if stable:
            ekf.P = (jnp.identity(ekf.P.shape[0]) - K @ H) @ ekf.P @ (jnp.identity(ekf.P.shape[0]) - K @ H).T + K @ R_d @ K.T
        else:
            ekf.P = ekf.P - K @ H @ ekf.P

        return ekf

@partial(jax.jit, static_argnames=['h', 'stable'])
def update_nonlinear(ekf,z, h, R, U, dt, stable=False):
        # time update
        ekf = jax.lax.cond(dt > 0, __update, __nothing, ekf, U, dt)

        H = jax.jacrev(h, argnums=0)(ekf.X, U)
        R_d = discretize.discretizeR(R, ekf.last_dt)

        # measurement update
        R_e = R_d + H @ ekf.P @ H.T
        K = ekf.P @ H.T @ jnp.linalg.inv(R_e)
        e = z - h(ekf.X, U) # innovations of the KF

        ekf.X = ekf.X + K @ e
        if stable:
            ekf.P = (jnp.identity(ekf.P.shape[0]) - K @ H) @ ekf.P @ (jnp.identity(ekf.P.shape[0]) - K @ H).T + K @ R_d @ K.T
        else:
            ekf.P = ekf.P - K @ H @ ekf.P

        return ekf

        
def __predict(ekf, dt):
        IJ = jnp.identity(ekf.num_states) + dt * ekf.J(ekf.X.reshape((ekf.num_states,)), ekf.U)
        Q = discretize.discretizeQ(ekf.J(ekf.X.reshape((ekf.num_states,)), ekf.U), ekf.Q_c, ekf.num_states, dt)
        ekf.P = IJ @ ekf.P @ IJ.T + dt * Q
        ekf.P_predicted = ekf.P
        ekf.X = ekf.X + dt * ekf.F_c(ekf.X, ekf.U)
        ekf.X_predicted = ekf.X
        return ekf


