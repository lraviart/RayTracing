import numpy as np
import os
import csv
import json


### Ray-tracing related utilities ###

def format_cir(cir):

    """
    Extract the delays and magnitudes from the cir object given by sionna
    
    Args: 
        cir: tuple of (coeffs_re, coeffs_im), delays
        
    Returns:
        delays: list of arrays (for each receiver) of delays of the paths
        magnitudes: list of arrays (for each receiver) of magnitudes of the paths
    
    """

    coeffs_re, coeffs_im = cir[0]
    delays = cir[1]

    # To numpy arrays
    coeffs_re = coeffs_re[..., 0].numpy() # Number of time steps = 1
    coeffs_im = coeffs_im[..., 0].numpy()
    delays = delays[..., :].numpy()

    a = coeffs_re + 1j*coeffs_im
    a = a.reshape(a.shape[0], -1)
    tau = delays.reshape(delays.shape[0], -1)

    # Removing points corresponding to -1 delay (no path)
    a_ = a
    tau_ = tau
    a = []
    tau = []
    for i in range(a_.shape[0]):
        mask = tau_[i] >= 0
        a.append(a_[i][mask])
        tau.append(tau_[i][mask])

    # Sort paths by delay
    for i in range(len(tau)):
        sort_indices = np.argsort(tau[i])
        tau[i] = tau[i][sort_indices]
        a[i] = a[i][sort_indices]

    return a, tau



def format_paths(cir, angles, array=False):

    """
    Extract the delays, magnitudes and angles from the cir and angles objects given by sionna

    Args:
        cir: tuple of (coeffs_re, coeffs_im), delays
        angles: list of arrays of shape [num_rx, num_tx, num_paths] 

    Returns:
        delays: list of arrays (for each receiver) of delays of the paths
        magnitudes: list of arrays (for each receiver) of magnitudes of the paths
        angles: list of arrays (for each receiver) of angles of the paths
    
    """

    coeffs_re, coeffs_im = cir[0]
    delays = cir[1]

    # To numpy arrays
    coeffs_re = coeffs_re[..., 0].numpy() # Number of time steps = 1
    coeffs_im = coeffs_im[..., 0].numpy()
    delays = delays[..., :].numpy()
    angles = angles[..., :].numpy()

    a = coeffs_re + 1j*coeffs_im
    if not array:
        a = a.reshape(a.shape[0], -1) 
    else:
        a = a.reshape(a.shape[0], a.shape[1], -1)
    tau = delays.reshape(delays.shape[0], -1)
    angles = angles.reshape(angles.shape[0], -1)


    # Removing points corresponding to -1 delay (no path)
    a_ = a
    tau_ = tau
    angles_ = angles
    a = []
    tau = []
    angles = []
    for i in range(a_.shape[0]):
        mask = tau_[i] >= 0
        if not array:
            a.append(a_[i][mask])
        else:
            a_append = np.zeros((a_.shape[1], np.sum(mask)), dtype=complex)
            for j in range(a_.shape[1]):
                a_append[j] = a_[i, j][mask]
            a.append(a_append)
        tau.append(tau_[i][mask])
        angles.append(angles_[i][mask])

    # Sort paths by delay
    for i in range(len(tau)):
        sort_indices = np.argsort(tau[i])
        if not array:
            a[i] = a[i][sort_indices]
        else:
            a_append = np.zeros((a[i].shape[0], a[i].shape[1]), dtype=complex)
            for j in range(a_.shape[1]):
                a_append[j] = a[i][j][sort_indices]
            a[i] = a_append
        tau[i] = tau[i][sort_indices]
        angles[i] = angles[i][sort_indices]

    return a, tau, angles
    

### Angle utilities ###

def bound_angle(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

def local_angle(global_angle, RX_orientation):
    return bound_angle(global_angle - RX_orientation[0])

def global_angle(local_angle, RX_orientation):
    return bound_angle(local_angle + RX_orientation[0])


def format_local_angles(angles, RX_orientations):

    """
    Format the angles given by sionna to be relative to the receiver orientation

    Args:
        angles: list of arrays of shape [num_rx, num_tx, num_paths] 
        RX_orientations: list of arrays of shape [num_rx, 3] (yaw, pitch, roll)

    Returns:
        angles: list of arrays (for each receiver) of angles of the paths relative to the receiver orientation
    
    """
    local_angles = []
    for i in range(len(angles)):
        local_angles.append(local_angle(angles[i], RX_orientations[i]))

    return local_angles



### Taps to csv file for plots ###


# A small helper class to handle NumPy arrays inside json.dumps
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def write_taps_to_csv(trial_id, delays, angles, tx_location, rx_positions, rx_orientations, filename):
    """
    Write paths data to a CSV, including Tx location, Rx positions, 
    and Rx orientations as metadata in the first three lines.
    """
    filepath = os.path.join("plot_csv", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    file_exists = os.path.isfile(filepath)

    with open(filepath, 'a', newline='') as f:
        if not file_exists:
            # Added cls=NumpyEncoder to safely handle any np.array passed here
            f.write(f"# Tx Location: {json.dumps(tx_location, cls=NumpyEncoder)}\n")
            f.write(f"# Rx Positions: {json.dumps(rx_positions, cls=NumpyEncoder)}\n")
            f.write(f"# Rx Orientations: {json.dumps(rx_orientations, cls=NumpyEncoder)}\n")
            
            writer = csv.writer(f)
            writer.writerow(["trial_id", "receiver", "path_index", "delay", "angle"])
        else:
            writer = csv.writer(f)
            
        for rx_idx in range(len(delays)):
            num_taps = len(delays[rx_idx])
            for tap_idx in range(num_taps):
                writer.writerow([
                    trial_id, 
                    rx_idx, 
                    tap_idx, 
                    delays[rx_idx][tap_idx], 
                    angles[rx_idx][tap_idx]
                ])

def read_taps_from_csv(filename):
    """
    Read the metadata (Tx location, Rx positions/orientations) and tap data.
    """
    filepath = os.path.join("plot_csv", filename)
    raw_data = {}
    
    with open(filepath, 'r') as f:
        # 1. Read and parse the first THREE lines
        line1 = f.readline().strip()
        line2 = f.readline().strip()
        line3 = f.readline().strip()
        
        tx_location = json.loads(line1.replace("# Tx Location: ", ""))
        rx_positions = json.loads(line2.replace("# Rx Positions: ", ""))
        rx_orientations = json.loads(line3.replace("# Rx Orientations: ", ""))
        
        num_receivers = len(rx_positions)
        
        # 2. Read the rest of the file
        reader = csv.DictReader(f)
        for row in reader:
            trial = row['trial_id']
            rx = int(row['receiver'])
            delay = float(row['delay'])
            angle = float(row['angle'])
            
            if trial not in raw_data:
                raw_data[trial] = {'delays': {}, 'angles': {}}
            if rx not in raw_data[trial]['delays']:
                raw_data[trial]['delays'][rx] = []
                raw_data[trial]['angles'][rx] = []
                
            raw_data[trial]['delays'][rx].append(delay)
            raw_data[trial]['angles'][rx].append(angle)

    # 3. Reconstruct the arrays
    trials_data = {}
    for trial, data in raw_data.items():
        trial_delays = [[] for _ in range(num_receivers)]
        trial_angles = [[] for _ in range(num_receivers)]
        
        for rx in data['delays']:
            trial_delays[rx] = data['delays'][rx]
            trial_angles[rx] = data['angles'][rx]
            
        trials_data[trial] = {
            'delays': [np.array(d) if len(d) > 0 else [] for d in trial_delays],
            'angles': [np.array(a) if len(a) > 0 else [] for a in trial_angles]
        }
        
    return tx_location, rx_positions, rx_orientations, trials_data

    
def clear_taps_file(filename):
    """Deletes the previous CSV file if it exists so we start fresh."""
    filepath = os.path.join("plot_csv", filename)
    
    if os.path.isfile(filepath):
        os.remove(filepath)
        print(f"Cleared previous data: {filepath}")
    else:
        print(f"No existing file found to clear: {filepath}")