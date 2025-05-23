import pmcx
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pickle

# speed of light: 
n = 1.370
c = 2.998e+10
c = c / n # cm/s


def run_mcx(ua, us, g=0.85, n=1.370, distance = 15, tend =1e-08, devf = 10000, nphoton = 1e8):
    
    # define structure properties
    prop = np.array([
        [0.0, 0.0, 1.0, 1.0], # air
        [ua, us, g, n], #
    ])

    # define voxel matrix properties 
    vol = np.ones([100, 100, 100], dtype='uint8')
    vol[:, :, 0:1] = 0
    vol[:, :, 1:] = 1

    # define the boundary:
    vol[:, :, 0] = 0
    vol[:, :, 99] = 0
    vol[0, :, :] = 0
    vol[99, :, :] = 0
    vol[:, 0, :] = 0
    vol[:, 99, :] = 0
    
    cfg = {
          'nphoton': nphoton,
          'vol': vol,
          'tstart': 0, # start time = 0
          'tend': tend, # end time
          'tstep': tend/devf, # step size
          'srcpos': [50, 50, 1],
          'srcdir': [0, 0, 1],  # Pointing toward z=1
          'prop': prop,
          'detpos': [[50+distance, 50, 1, 2]], 
          'savedetflag': 'dpxsvmw',  # Save detector ID, exit position, exit direction, partial path lengths
          'unitinmm': 1,
          'autopilot': 1,
          'debuglevel': 'DP',

          # Isotropic source:
          'srctype': 'isotropic',
          #'tmod': mf,
    }
    cfg['issaveref']=1
    cfg['issavedet']=1
    cfg['issrcfrom0']=1
    cfg['maxdetphoton']=nphoton

    # Run the simulation
    res = pmcx.mcxlab(cfg)
    #print(res['stat'])
    #print("Result keys:", list(res.keys()))
    #print('detp keys:', res['detp'].keys())
    return res, cfg

# sum_dref_per_time = x weights/mm-1/photon
def get_intensity_dynamic(cfg, res):
    # get mask. 
    det_x, det_y, det_z, det_r = cfg['detpos'][0]
    nx, ny, nz, nt = res['dref'].shape
    x_grid, y_grid = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    det_mask = (x_grid - det_x)**2 + (y_grid - det_y)**2 <= det_r**2
    
    # get intensity dynamic
    sum_dref_per_time = []
    # loop each t 
    for t in range(res['dref'].shape[3]):
        dref_slice = res['dref'][:, :, 0, t]
        sum_val = np.sum(dref_slice[det_mask])
        sum_dref_per_time.append(sum_val)
    return sum_dref_per_time




def mcx_simulation(ua, us, g=0.85, n=1.370, distance = 15, tend =1e-08, devf = 10000, nphoton = 1e8):
    """
    Wrapper function to run MCX simulation and extract time-resolved intensity.

    Parameters:
        ua (float): Absorption coefficient [1/mm]
        us (float): Scattering coefficient [1/mm]
        g (float): Anisotropy factor (default: 0.85)
        n (float): Refractive index (default: 1.370)
        distance (float): Source-detector separation in mm (default: 15)
        tend (float): Simulation time duration (default: 1e-08 s)
        devf (int): Number of time intervals (default: 10000)
        nphoton (float): Total number of photons (default: 1e8)

    Returns:
        intensity_d (list): Time-resolved detector intensity values
        unit (float): Time step per frame (i.e., temporal resolution)
    """
    res, cfg = run_mcx(ua, us, g, n, distance, tend, devf, nphoton)
    intensity_d = get_intensity_dynamic(cfg, res)
    t = np.linspace(0, tend, devf)
    intensity_d /= np.trapz(intensity_d, t) # normalize the signal such that the AUC == 1. 
    unit = tend / devf
    return intensity_d, unit

# get the fft, freqs. 
def mcx_fft(ua, us, g=0.85, n=1.370, distance = 15, tend =1e-08, devf = 10000, nphoton = 1e8):
    intensity_d, unit = mcx_simulation(ua, us, g, n, distance, tend, devf, nphoton)
    # adjust to weight/bin: 
    intensity_d = np.array(intensity_d) # * nphoton * (tend/devf)
    fft_result = np.fft.fft(intensity_d) # a+bj
    freqs = np.fft.fftfreq(intensity_d.shape[0], unit)
    return fft_result, freqs


# return uac, udc and phase from fft results.  
def extract_freq(target_freq, TPSF, tend, devf):
    
  # proposed method: 
    t = np.linspace(0, tend, devf)
    omega = 2 * np.pi * target_freq # 2*pi*f 
    I_f = np.trapz(TPSF * np.exp(-1j * omega * t), t)  # Complex integral
    amplitude = np.abs(I_f)
    phase_deg = np.angle(I_f, deg=True)  # Phase in degrees
    udc = np.trapz(TPSF, t) # U_dc
    return amplitude, udc, phase_deg

def extract_freq2(target_freq, freqs, fft_result, tend, devf):
    N = len(freqs)
    dt = tend/ devf

    # Find the closest frequency index
    index = np.argmin(np.abs(freqs - target_freq))

    # DC component (0 Hz)
    udc = np.real(fft_result[0]) / N

    # Amplitude at target frequency (normalize for one-sided spectrum)
    uac = np.abs(fft_result[index]) / (N / 2)
    
    # Phase at target frequency in degrees
    phase_deg = np.angle(fft_result[index], deg=True)

    return uac, udc, phase_deg




