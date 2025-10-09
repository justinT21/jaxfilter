import jax
import jax.numpy as jnp
import discretize

class Rts:
    def __init__(self, F_c, num_states:int) -> None:
       self.num_states = num_states
       self.F_c = F_c
       self.J = jax.jacrev(self.F_c, argnums = 0)

    def run(self, time_series, X_filtered, X_predicted, P_filtered, P_predicted, U_series, Q_c, stable=False):
        def reverse_loop(carry, x):
            future_t, future_X_rts, future_P_rts = carry
            t, X_filtered, X_predicted, P_filtered, P_predicted, U_control = x
            dt = future_t - t

            A = jnp.identity(self.num_states) + dt * self.J(X_filtered.reshape((self.num_states,)), U_control)
            S = P_filtered @ A.T @ jnp.linalg.inv(P_predicted)
            if stable :
               delta = X_filtered - S @ X_predicted
               Q = discretize.discretizeQ(A, Q_c, A.shape[0], dt)
               X_rts = S @ future_X_rts + delta
               P_rts = S @ (future_P_rts + Q) @ S.T + (jnp.identity(P_filtered.shape[0]) - S @ A) @ P_filtered @ (jnp.identity(P_filtered.shape[0]) - S @ A).T
            else:
                X_rts = X_filtered + S @ (future_X_rts - X_predicted)
                P_rts = P_filtered - S @ (P_predicted - future_P_rts) @ S.T

            return (t, X_rts, P_rts), (X_rts, P_rts)


        _, (X_rts, P_rts) = jax.lax.scan(f=reverse_loop, init=(time_series[-1], X_filtered[-1], P_filtered[-1]), xs=(time_series[:-1], X_filtered[:-1], X_predicted[1:], P_filtered[:-1], P_predicted[1:], U_series[:-1]), reverse=True)
        return X_rts, P_rts

    def filter_and_run(self, time_series, X_data, U_data, h, R, ekf, stable=False):
        def loop(carry, x):
            filter, previous_t = carry
            t, X_measured, U = x
            dt = t - previous_t

            filter.update_nonlinear(X_measured, h, R, U, dt, stable = stable)
            X_predict = filter.X 

            return (filter, t), (X_predict, filter.X_predicted, filter.P, filter.P_predicted)


        _, (X_filtered, X_predicted, P_filtered, P_predicted) = jax.lax.scan(f=loop, init=filter, xs=(time_series, X_data, U_data))

        X_rts,_ = self.run(time_series, X_filtered, X_predicted, P_filtered, P_predicted, U_data, ekf.Q_c, stable)

        return X_rts
        
