

import numpy as np
rng = np.random.default_rng(seed=42)
import pandas as pd
from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sqlite3


def get_cap_points(dom_spacing, r_bounds):

    num_pts = 2.3902220858625336 / (dom_spacing/r_bounds)**2

    indices = np.arange(0, num_pts, dtype=float) + 0.5

    r = r_bounds * np.sqrt(indices/num_pts)
    theta = np.pi * (1 + 5**0.5) * indices

    x = r*np.cos(theta)
    y = r*np.sin(theta)

    return x, y


def get_dom_points(h, r, dom_spacing):

    # Sides
    n_points = int(np.floor(2*np.pi*r / dom_spacing))
    thetas = np.linspace(-np.pi, np.pi, n_points, endpoint=False)
    x, y = r*np.cos(thetas), r*np.sin(thetas)

    n_points_z = int(np.floor(h / dom_spacing))
    z = np.linspace(-h/2, h/2, n_points_z)

    x = np.repeat(x[np.newaxis,:], n_points_z, axis=0)
    y = np.repeat(y[np.newaxis,:], n_points_z, axis=0)
    z = np.repeat(z[:,np.newaxis], n_points, axis=1)
    side_points = np.stack([x,y,z]).reshape(3, -1)

    # Top 
    x, y = get_cap_points(dom_spacing, r)
    z = np.zeros(len(x)) + h/2
    top_points = np.array([x, y, z])

    # Bottom
    x, y = get_cap_points(dom_spacing, r)
    z = np.zeros(len(x)) - h/2
    bottom_points = np.array([x, y, z])

    # All points
    all_points = np.concatenate(
        [
            side_points,
            top_points,
            bottom_points,
        ],
        axis=1
    )

    return all_points.T


def sim_events(n_events, h, r, dom_points):

    # Create interactions
    detector_size = np.linalg.norm([h,r])

    x_pos = rng.uniform(-r/2, r/2, n_events)
    y_pos = rng.uniform(-r/2, r/2, n_events)
    z_pos = rng.uniform(-h/2, h/2, n_events)
    interaction_points = np.stack([x_pos, y_pos, z_pos]).T

    # Create DOM hits
    distances = distance_matrix(interaction_points, dom_points)
    hits = distances < rng.exponential(detector_size/10, distances.shape)
    
    event_nos = np.array([np.arange(n_events)]*len(dom_points)).T
    pulsemap = np.append(np.array([dom_points]*n_events), event_nos[:,:,np.newaxis], axis=2)
    
    pulsemap = np.append(pulsemap, hits[:,:,np.newaxis], axis=2).reshape(-1, 5)

    pulsemap = pd.DataFrame(pulsemap, columns=['fX','fY', 'fZ', 'event_no', 'hit'])
    pulsemap.drop(pulsemap[pulsemap['hit']<.5].index, inplace=True)
    pulsemap.drop(['hit'], axis=1, inplace=True)

    pulsemap['fTime'] = rng.random(len(pulsemap))
    pulsemap['fCharge'] = rng.random(len(pulsemap))

    # Create truth values
    truth = pd.DataFrame(interaction_points, columns=['fVx','fVy', 'fVz'])
    truth['event_no'] = np.arange(n_events)
    truth['pid'] = rng.choice([12, 14], n_events)
    truth['Lpid'] = truth['pid'] - 1
    truth['interaction_type'] = rng.choice([1, 2], n_events)

    truth['fNdx'] = np.zeros(n_events)
    truth['fNdy'] = np.zeros(n_events)
    truth['fNdz'] = np.zeros(n_events) + 1
    truth['fLdx'] = rng.random(n_events)
    truth['fLdy'] = rng.random(n_events)
    truth['fLdz'] = rng.random(n_events)

    truth['fNE'] = rng.random(n_events)
    truth['fLE'] = 0.9 * truth['fNE']

    truth['fNphotons'] = rng.binomial(3, p=0.05)
    truth['fNpions'] = rng.binomial(3, p=0.05)

    return truth, pulsemap


def sim_reco(truth):

    # Simulate model reco
    model = truth.filter(['event_no'], axis=1)
    model['v_u'] = (truth['pid'] - 12)/2

    scale_factors = rng.exponential(1, len(truth))
    model['v_u_pred'] = model['v_u'] + (0.5-model['v_u']) * scale_factors/(np.max(scale_factors))
   
    # Simulate fiTQun reco
    fiTQun = truth.filter(['event_no'], axis=1)
    fiTQun['fqmu_nll'] = 50 - (truth['pid'] - 12)/2 * 50 + rng.exponential(50, len(truth))
    fiTQun['fqe_nll'] = (truth['pid'] - 12)/2 * 50 + rng.exponential(50, len(truth))                      

    return model, fiTQun


def run_sql_code(database_path, code):

    # Conenct and run code
    conn = sqlite3.connect(database_path)
    c = conn.cursor()
    c.executescript(code)
    c.close()


def insert_data(database_path, table_name, data):
    with sqlite3.connect(database_path) as conn:
        data.to_sql(table_name, conn, if_exists='replace',index=False)
        # conn.commit()
        # conn.close()


def export_sim(truth, pulsemap, model, fiTQun, database_name, data_dir='test_data/'):

    # Make database
    db_path = data_dir+database_name+'.db'
    results_path = data_dir+database_name+'_reco.csv'

    truth_sql = """
        CREATE TABLE IF NOT EXISTS truth(
        event_no   INTEGER PRIMARY KEY NOT NULL, 
        particle_sign      INTEGER, 
        interaction_type   INTEGER, 

        fVx        FLOAT, 
        fVy        FLOAT, 
        fVz        FLOAT, 

        pid        INTEGER NOT NULL, 
        fNE        FLOAT, 
        fNdx       FLOAT, 
        fNdy       FLOAT, 
        fNdz       FLOAT, 

        Lpid       INTEGER NOT NULL, 
        fLE        FLOAT, 
        fLdx       FLOAT, 
        fLdy       FLOAT, 
        fLdz       FLOAT, 

        fNphotons  INTEGER, 
        fNpions    INTEGER );
    """

    fitqun_sql = """
        CREATE TABLE IF NOT EXISTS fiTQun(
        event_no   INTEGER PRIMARY KEY NOT NULL, 

        fqe_nll    FLOAT, 
        fqmu_nll   FLOAT );
    """

    pulsemap_sql = """
        CREATE TABLE IF NOT EXISTS PULSES(
        event_no   INTEGER NOT NULL, 
        fX         FLOAT, 
        fY         FLOAT, 
        fZ         FLOAT, 
        fTime      FLOAT, 
        fCharge    FLOAT );
      """

    # Add truth
    run_sql_code(db_path, truth_sql)
    insert_data(db_path, 'truth', truth)

    # Add pulsemap
    run_sql_code(db_path, pulsemap_sql)
    insert_data(db_path, 'PULSES', pulsemap)
    run_sql_code(db_path, 'CREATE INDEX event_no_PULSES ON PULSES (event_no);')

    # Add fiTQun
    run_sql_code(db_path, fitqun_sql)
    insert_data(db_path, 'fiTQun', fiTQun)

    # Export model
    model.to_csv(results_path)

# Generate database

# Set detector parameters
h = 600
r = 200
dom_spacing = 20

# Generate dom locations
dom_points = get_dom_points(h, r, dom_spacing)

# Simulate events
n_events = 100

truth, pulsemap = sim_events(n_events, h, r, dom_points)

# Simulate reconstruction
model, fiTQun = sim_reco(truth)

# Export simulation
export_sim(truth, pulsemap, model, fiTQun, 'test_database')

