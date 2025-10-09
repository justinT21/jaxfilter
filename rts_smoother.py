import jax
import jax.numpy as jnp

class Rts:
    def __init__(self, F_c, num_states:int) -> None:
       self.num_states = num_states
       self.F_c = F_c
       self.J = jax.jacrev(self.F_c, argnums = 0)
        pass

    def run(self, time_series, X_filtered, X_predicted, P_filtered, P_predicted, U_series):

        def reverse_loop(carry, x):
            future_t, future_X_rts, future_P_rts = carry
            t, X_filtered, X_predicted, P_filtered, P_predicted, U_control = x
            dt = future_t - t

            A = jnp.identity(self.num_states) + dt * self.J(X_filtered.reshape((self.num_states,)), U_control)
            S = P_filtered @ A.T @ jax.linalg.inv(P_predicted)
            X_rts = X_filtered + S @ (future_X_rts - X_predicted)
            P_rts = P_filtered - S @ (P_predicted - future_P_rts) @ S.T

            return (t, X_rts, P_rts), (X_rts, P_rts)


        X_rts, P_rts = jax.lax.scan(f=reverse_loop, init=(time_series[-1], X_filtered[-1], P_filtered[-1]), x=(time_series[:-1], X_filtered[:-1], X_predicted[1:], P_filtered[:-1], P_predicted[1:], U_series[:-1]), reverse=True)

    def filter_and_run(self, time_series, ekf):
