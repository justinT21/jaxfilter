import os

os.environ['JAX_PLATFORMS'] = 'cuda,cpu'

from absl.testing import absltest
import jax.numpy as jnp
import jax
import ekf
import rts_smoother
import matplotlib.pyplot as plt


class EKFTest(absltest.TestCase):
    def setUp(self) -> None:
       self.key = jax.random.key(0)
       self.num_steps = 100
       self.dt = 0.1

    def test_linear_A(self):
        # x = [x_level, x_flow].T
        A = jnp.array([[0, 1.],[0, 0]])
        H = jnp.array([[1., 0]])
        Q = jnp.diag(jnp.array([0, 0.00001]))
        X_initial = jnp.array([[0., 0]]).T
        P_initial = jnp.diag(jnp.array([1000., 1000.]))
        U_initial = jnp.empty((0,))
        R = jnp.array([[0.1]])
        filter = ekf.Ekf(F_c=lambda X, U: A @ X, Q_c=Q, X_initial=X_initial, U_initial = U_initial, P_initial=P_initial, num_states=2, num_control=0)

        self.key, first_subkey = jax.random.split(self.key)
        def loop(carry, x):
            key, previous_X, previous_t, filter = carry
            key, subkey = jax.random.split(key)
            t = previous_t + self.dt

            X_truth = previous_X + jnp.array([[0.5, 0.]]).T * self.dt
            X_measured = X_truth[0,0] + jax.random.normal(subkey, shape=(1,1)) * 0.3 * self.dt
            filter.update_linear(X_measured , H, R, U_initial, self.dt, stable=True)
            X_predict = filter.X

            return (key, X_truth, t, filter), (X_truth.reshape(2,), t, X_measured, X_predict.reshape(2,), filter.X_predicted.reshape(2,), filter.P, filter.P_predicted)

        _, series_values = jax.lax.scan(f=loop, init=(first_subkey, X_initial, 0.0, filter), length = self.num_steps)

        smoother = rts_smoother.Rts(filter.F_c, filter.num_states)
        print(series_values[3].shape)
        print(jnp.empty((series_values[1].shape[0],0,)).shape)
        X_rts,_ =smoother.run(series_values[1], series_values[3], series_values[4], series_values[5], series_values[6], jnp.empty((series_values[1].shape[0],0,)), filter.Q_c, stable=True)

        plt.plot(series_values[1], series_values[0][:, 0], label = "truth") 
        plt.plot(series_values[1], series_values[2][:, 0, 0], label = "measured") 
        plt.plot(series_values[1], series_values[3][:, 0], label = "kf") 
        plt.plot(series_values[1], series_values[3][:, 1], label = "kf-flow") 
        print(X_rts.shape)
        plt.plot(series_values[1][:-1], X_rts[:, 0], label = "kf-rts") 
        plt.plot(series_values[1][:-1], X_rts[:, 1], label = "flow-rts") 
        plt.legend()
        #plt.show()
        print(jnp.max(jnp.abs(series_values[0][:-1,0] - X_rts[:, 0])))
        print(jnp.max(jnp.abs(series_values[0][:,0] - series_values[3][:, 0])))

        self.assertTrue(jnp.allclose(series_values[0][:-1, 0], X_rts[:, 0], atol=0.002))
        self.assertTrue(jnp.allclose(series_values[0][:, 0], series_values[3][:, 0], atol=0.05))

if __name__ == '__main__':
  absltest.main()
