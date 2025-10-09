import jax.numpy as jnp
import jax
import discretize

@jax.tree_util.register_pytree_node_class
class Ekf:
    def __init__(self, F_c, Q_c, X_initial, U_initial, P_initial, num_states: int, num_control: int) -> None:
       self.X = X_initial
       self.P = P_initial
       self.U = U_initial
       self.F_c = F_c
       self.Q_c = Q_c
       self.num_states = num_states
       self.num_control = num_control
       self.J = jax.jacrev(self.F_c, argnums = 0)

    def tree_flatten(self):
        return ((self.X, self.U, self.P), (self.F_c, self.Q_c, self.J, self.num_states, self.num_control))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        ekf= object.__new__(Ekf)
        ekf.X, ekf.U, ekf.P = children
        ekf.F_c, ekf.Q_c, ekf.J, ekf.num_states, ekf.num_control = aux_data
        return ekf


    def update_linear(self,z, H, R, U, dt, stable=False) -> None:
        # time update
        if dt > 0:
            self.U = U
            self.__predict(dt)

        R_d = discretize.discretizeR(R, dt)

        # measurement update
        R_e = R_d + H @ self.P @ H.T
        K = self.P @ H.T @ jnp.linalg.inv(R_e)
        e = z - H @ self.X # innovations fo the KF

        self.X = self.X + K @ e
        if stable:
            self.P = (jnp.identity(self.P.shape[0]) - K @ H) @ self.P @ (jnp.identity(self.P.shape[0]) - K @ H).T + K @ R_d @ K.T
        else:
            self.P = self.P - K @ H @ self.P

    def update_nonlinear(self,z, h, R, U, dt, stable=False) -> None:
        # time update
        if dt > 0:
            self.U = U
            self.__predict(dt)

        H = jax.jacrev(h)(self.X)
        R_d = discretize.discretizeR(R, dt)

        # measurement update
        R_e = R_d + H @ self.P @ H.T
        K = self.P @ H.T @ jnp.linalg.inv(R_e)
        e = z - h(self.X) # innovations of the KF

        self.X = self.X + K @ e
        if stable:
            self.P = (jnp.identity(self.P.shape[0]) - K @ H) @ self.P @ (jnp.identity(self.P.shape[0]) - K @ H).T + K @ R_d @ K.T
        else:
            self.P = self.P - K @ H @ self.P

        
    def __predict(self, dt):
        IJ = jnp.identity(self.num_states) + dt * self.J(self.X.reshape((self.num_states,)), self.U)
        Q = discretize.discretizeQ(self.J(self.X.reshape((self.num_states,)), self.U), self.Q_c, self.num_states, dt)
        self.P = IJ @ self.P @ IJ.T + dt * Q
        self.P_predicted = self.P
        self.X = self.X + dt * self.F_c(self.X, self.U)
        self.X_predicted = self.X


