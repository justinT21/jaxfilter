import jax.numpy as jnp
import jax

def discretizeAB(A, B, num_states, num_control, dt):
    G = jax.scipy.linalg.expm(dt * jnp.block([[A, B], [jnp.zeros((num_control, num_states + num_control))]]))
    A_d = G[:num_states, :num_states]
    B_d = G[:num_states, num_states:]
    return A_d, B_d 

def discretizeQ(A, Q, num_states, dt):
    F = dt * jnp.block([[-A, Q], [jnp.zeros((num_states, num_states)), A.T]])
    G = jax.scipy.linalg.expm(F)
    Q_d = (G[num_states:, num_states:]).T @ G[:num_states, num_states:]
    return Q_d

def discretizeR(R, dt):
    return R * 1 / dt
