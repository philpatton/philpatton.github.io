
from jax import random
from jax.scipy.special import logit
from numpyro.contrib.control_flow import scan
from numpyro.infer import NUTS, MCMC, Predictive
import arviz as az
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import seaborn as sns

# plotting defaults
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = False
sns.set_palette("tab10")

# hyper parameters
BUFFER = 100
M = 200
U_SIGMA = 150
RANDOM_SEED = 17

# mcmc hyperparameters
CHAIN_COUNT = 1
WARMUP_COUNT = 500
SAMPLE_COUNT = 1000

def get_oven_data():

    # coordinates for each trap
    ovenbird_trap = pd.read_csv('ovenbirdtrap.txt', delimiter=' ')

    # information about each trap
    trap_coords = ovenbird_trap[['x', 'y']].to_numpy()

    # bounds of the state space
    xmin, ymin = trap_coords.min(axis=0) - BUFFER
    xmax, ymax = trap_coords.max(axis=0) + BUFFER

    # ovenbird capture history
    oven_ch = pd.read_csv('ovenbirdcapt.txt', delimiter=' ')

    df = oven_ch

    # Get unique values for each dimension
    years = df['Year'].unique()
    bands = df['Band'].unique()
    nets = ovenbird_trap['Net'].unique()
    days = df['Day'].unique()

    # Create mapping dictionaries for fast lookup
    year_to_idx = {year: i for i, year in enumerate(years)}
    band_to_idx = {band: i for i, band in enumerate(bands)}
    net_to_idx = {net: i for i, net in enumerate(nets)}
    day_to_idx = {day: i for i, day in enumerate(days)}

    # Initialize 4D array
    y_spatial = np.zeros((len(years), len(bands), len(nets), len(days)), dtype=np.int8)

    # Vectorized indexing
    band_indices = df['Band'].map(band_to_idx).values
    net_indices = df['Net'].map(net_to_idx).values
    day_indices = df['Day'].map(day_to_idx).values
    year_indices = df['Year'].map(year_to_idx).values

    y_spatial[year_indices, band_indices, net_indices, day_indices] = 1
    y = y_spatial.sum(axis=(2, 3))

    # augment y
    all_zero_history = np.zeros((len(years), M - len(bands)), dtype=np.int8)
    y_augmented = np.hstack((y, all_zero_history))

    # augment y spatial
    all_zero_history = np.zeros(
        (len(years), M - len(bands), len(nets), len(days)), dtype=np.int8
    )
    y_spatial_augmented = np.concatenate((y_spatial, all_zero_history), axis=1)

    occasion_count = (
        df[['Year', 'Day']].drop_duplicates()
            .groupby('Year')
            .count()
            # .to_dict()['Day']
            .values[:, 0]
    )

    return {
        'y_spatial': y_spatial_augmented,
        'y': y_augmented,
        'trapxy': trap_coords,
        'trap_count': len(trap_coords),
        'occasion_count': occasion_count,
        'minima': (xmin, ymin),
        'maxima': (xmax, ymax),
        'season_count': len(years),
        'super_size': M,
    }

oven_data = get_oven_data()

def spatial_js(data):

    # unpack everything in the data dictionary
    history = data['y_spatial'].sum(axis=3)
    trapxy = data['trapxy']
    trap_count = data['trap_count']
    xmin, ymin = data['minima']
    xmax, ymax = data['maxima']
    super_size = data['super_size']
    occasion_count = jnp.array(data['occasion_count'])
    season_count = data['season_count']

    # transition probabilities
    phi = numpyro.sample('phi', dist.Uniform(0, 1))
    with numpyro.plate('intervals', season_count):
        gamma = numpyro.sample('gamma', dist.Uniform(0, 1))

    # parameters related to detection
    g0 = numpyro.sample('g0', dist.Uniform(0, 1))
    sigma = numpyro.sample('sigma', dist.Uniform(0, U_SIGMA))

    # functions of the natural parameters
    alpha0 = logit(dist.util.clamp_probs(g0))
    alpha1 = 1 / (2 * sigma**2)

    def transition_and_capture(carry, y_current):

        z_previous, t = carry

        # transition probability matrix
        trans_probs = jnp.array([
            [1 - gamma[t], gamma[t],     0.0],  # From not yet entered
            [         0.0,      phi, 1 - phi],  # From alive
            [         0.0,      0.0,     1.0]   # From dead
        ])

        with numpyro.plate("animals", super_size, dim=-1):

            mu_z_current = trans_probs[z_previous]
            z_current = numpyro.sample(
                "state",
                dist.Categorical(dist.util.clamp_probs(mu_z_current)),
                infer={"enumerate": "parallel"}
            )

            # sample each coordinate seperately then stack
            sx = numpyro.sample('sx', dist.Uniform(xmin, xmax))
            sy = numpyro.sample('sy', dist.Uniform(ymin, ymax))
            center_coords = jnp.stack([sx, sy], axis=1)

            # pairwise distance between activity centers and the traps
            distance = jnp.linalg.norm(
                center_coords[:, None, :] - trapxy[None, :, :],
                axis=2
            )

            # cloglog version
            lm = jnp.exp(alpha0 - alpha1*distance**2)
            p = 1 - jnp.exp(-lm)

            mu_y_current = jnp.where(z_current == 1, p.T, 0.0)
            mu_y_current = dist.util.clamp_probs(mu_y_current)
            with numpyro.plate('traps', trap_count):
                numpyro.sample(
                    "obs",
                    dist.Binomial(occasion_count[t], mu_y_current),
                    obs=y_current
                )

        return (z_current, t + 1), None

    # start everyone in the not yet entered state
    state_init = jnp.zeros(super_size, dtype=jnp.int32)
    scan(
        transition_and_capture,
        (state_init, 0),
        history
    )

rng_key = random.PRNGKey(RANDOM_SEED)

# specify which sampler you want to use
nuts_kernel = NUTS(spatial_js)

# configure the MCMC run
mcmc = MCMC(nuts_kernel, num_warmup=WARMUP_COUNT, num_samples=SAMPLE_COUNT,
            num_chains=CHAIN_COUNT)

# run the MCMC then inspect the output
mcmc.run(rng_key, oven_data)
mcmc.print_summary()