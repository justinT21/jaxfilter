import ekf
import jax

def run_nonlinear_on_array(F, h, R, Q, t_data, X_measured, U_data, P_initial, num_states, num_control):
    filter = ekf.Ekf(F_c=F, Q_c=Q, X_initial=X_measured[0], U_initial=U_data[0],P_initial=P_initial, num_states=num_states, num_control=num_control)

    def loop(carry, x):
        filter, previous_t = carry
        t, X_measured, U = x
        dt = t - previous_t

        filter.update_nonlinear(X_measured, h, R, U, dt)
        X_predict = filter.X 

        return (filter, t), X_predict


    _, series_values = jax.lax.scan(f=loop, init=filter, xs=(t_data, X_measured, U_data))

    return series_values

