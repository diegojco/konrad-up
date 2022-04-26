import sys
import os
import os.path
import datetime
import numpy as np
from netCDF4 import Dataset as ncload
import scipy.interpolate as interp
import konrad
import traceback

# It sets the main output directory
output_dir = "/work/mh0066/m300556/konrad_exps/output_files/"

# It sets the source directory
konrad_dir = "/work/mh0066/m300556/konrad_sandbox/konrad/"

# It sets default values
O3 = "a"
cnv = "h"
upw = 0.2
CO2 = 1.0
RH = 0.4
is_cntrl2 = False
is_cntrl = True
user_cntrl = ""


# It reads arguments from standard input
if len(sys.argv) > 1:
    O3 = str(sys.argv[1])
if len(sys.argv) > 2:
    cnv = str(sys.argv[2])
if len(sys.argv) > 3:
    upw = float(sys.argv[3])
if len(sys.argv) > 4:
    CO2 = float(sys.argv[4])
if len(sys.argv) > 5:
    RH = float(sys.argv[5])
if len(sys.argv) > 6:
    is_cntrl = bool(int(sys.argv[6]))
if len(sys.argv) > 7:
    is_cntrl2 = bool(int(sys.argv[7]))
if len(sys.argv) > 8:
    user_cntrl = str(sys.argv[8])


# It validates options. If not valid, defaults are loaded
print("Validating settings...")
if O3 not in ["f", "a"]:
    print("Invalid ozone option: Switching to Cariolle")
    O3 = "a"
if cnv not in ["h"]:
    print("Invalid convection scheme: Switching to hard convective adjustment")
    cnv = "h"
if (
    (upw >= 1.0) and
    (upw not in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
) or upw < 0.0:
    print("Invalid stratospheric upwelling speed: Switching to w_{upw} = 0.2 mm/s")
    upw = 0.2
if RH < 0.0:
    print("Invalid RH profile: Switching to a 0.4 constant tropospheric RH")
    RH = 0.4
RHF = int(RH)
UTRH = RH - RHF
if RHF == 3:
    RH = 3.6
    UTRH = 0.6
if is_cntrl and is_cntrl2:
    print("Invalid control run: Switching to fixed SST. Switching off free SST")
    is_cntrl = True
    is_cntrl2 = False
if not os.path.isfile(output_dir + "_".join(user_cntrl.split(" ")) + ".nc"):
    print("Invalid starting experiment: Switching to the default control")
    user_cntrl = ""

# Heat sink corrections for upwelling effects and time step settings
timestep = 1
incr = 60
tmin = timestep
tmax = 7200
maxdur = 365 * 10
etol = 1e-10
correction = 0.0
if is_cntrl:
    maxdur = 365 * 5 
    etol = 1e-20
if is_cntrl2:
    if RH == 0.4:
        correction = 3.50224609375 # 3.5021484375 3.50234375
    elif RH == 2.8:
        correction = 3.4048828125 # 3.4046875 3.405078125
    elif RH == 0.7:
        correction = 3.39843046875 # 3.398353125 3.3985078125
    elif RH == 3.6:
        correction = 3.4634765625 # 3.46328125 3.463671875

print(timestep, incr, tmin, tmax, etol, correction)

# It transforms numeric options to strings
upw_l = f"{int(upw * 1000):03d}"
if upw == 1.0:
    upw_l = "p01"
elif upw == 2.0:
    upw_l = "p02"
elif upw == 3.0:
    upw_l = "p03"
elif upw == 4.0:
    upw_l = "p04"
elif upw == 5.0:
    upw_l = "p05"
elif upw == 6.0:
    upw_l = "p06"
elif upw == 7.0:
    upw_l = "p07"
elif upw == 8.0:
    upw_l = "p08"
elif upw == 9.0:
    upw_l = "p09"
elif upw == 10.0:
    upw_l = "p10"
elif upw == 11.0:
    if O3 == "f":
        upw_l = "307"
    elif O3 == "a":
        upw_l = "298"
CO2_l = f"{int(CO2 * 10):03d}"
RH_l = f"{int(RH * 100):03d}"


# It creates the name of the control experiment, checking if file exists
if is_cntrl or is_cntrl2:
    cntrl_nam = f"conv-O3_last-{RH_l}-{CO2_l} {O3}{upw_l}{cnv} cntrl"
    test = os.path.isfile(output_dir + "_".join(cntrl_nam.split(" ")) + ".nc")
    if is_cntrl2 and not test:
        print("Missing Control experiment")
if (not is_cntrl) and (not is_cntrl2):
    if user_cntrl == "":
        cntrl_nam = f"conv-O3_last-{RH_l}-010 a200h picntrl"
        test = os.path.isfile(output_dir + "_".join(cntrl_nam.split(" ")) + ".nc")
        if not test:
            print("Missing pi-Control experiment")
    else:
        cntrl_nam = user_cntrl


# It creates the name of the experiment
if is_cntrl:
    base_name = cntrl_nam
elif is_cntrl2:
    base_name = cntrl_nam[:-5] + "picntrl"
else:
    base_name = f"conv-O3_last-{RH_l}-{CO2_l} {O3}{upw_l}{cnv}"
    if user_cntrl != "":
        index = 0
        test_str = base_name
        test = os.path.isfile(output_dir + "_".join(test_str.split(" ")) + ".nc")
        while test:
            index += 1
            test_str = base_name + f" alt_{index}"
            test = os.path.isfile(output_dir + "_".join(test_str.split(" ")) + ".nc")
        base_name = test_str


# It creates the output directory, if necessary
os.makedirs(output_dir, exist_ok=True)


# The Cariolle dataset: it obtains interpolations for initial conditions
cariolle_path = konrad_dir + "konrad/data/Cariolle_data.nc"
cariolle_data = ncload(cariolle_path, mode="r")
CARIOLLE_p = cariolle_data.variables["p"][:].data
CARIOLLE_T = cariolle_data.variables["A5"][:].data
CARIOLLE_O3 = cariolle_data.variables["A3"][:].data
CARIOLLE_Tp = interp.interp1d(CARIOLLE_p, CARIOLLE_T, 2)
CARIOLLE_O3p = interp.interp1d(CARIOLLE_p, CARIOLLE_O3, 2)


# Model Toolbox
## 1. Pressure levels (full and half), number of levels and null profile
plev, phlev = konrad.utils.get_pressure_grids(1000e2, 1, 500)
nlev = plev.size
null = lambda: np.array([np.zeros(nlev)])


## 2. Atmosphere (initialised with defaults and the above pressure levels)
atmo = lambda: konrad.atmosphere.Atmosphere(phlev=phlev)


## 3. Radiation (RRTMG)
radt = lambda: konrad.radiation.RRTMG()


## 4. Tropospheric humidity
### Default (constant)
def_tropo = lambda rhc: konrad.humidity.VerticallyUniform(rh_surface=rhc)
### Polynomial C-shaped
csp_tropo = lambda utrh: konrad.humidity.PolynomialCshapedRH(top_peak_rh=utrh)
### ERA5 C-shaped (Bourdin et al. 2020)
ERA_tropo = lambda: konrad.humidity.PolynomialCshapedRH(
    top_peak_T=None,
    top_peak_rh=0.5810173,
    freezing_pt_rh=0.3138377,
    bl_top_p=950e2,
    bl_top_rh=0.85981554,
    surface_rh= 0.85981554 + (-1.5954482555389405 * 50 * 1e-03),
)


## 5. Stratospheric humidity
### Default (coupled to the volume mixing ratio at the cold point)
def_strato = lambda: konrad.humidity.ColdPointCoupling()


## 6. Atmospheric humidity
### Default (Constant tropospheric humidity and cold-point coupling)
def_h = lambda rhc: konrad.humidity.FixedRH(
    rh_func=def_tropo(rhc), stratosphere_coupling=def_strato()
)
### Polynomial C-shaped
csp_h = lambda utrh: konrad.humidity.FixedRH(
    rh_func=csp_tropo(utrh), stratosphere_coupling=def_strato()
)
ERA_h = lambda: konrad.humidity.FixedRH(
    rh_func=ERA_tropo(), stratosphere_coupling=def_strato()
)


## 7. Stable lapse rates
### Default (Moist lapse-rate)
moist = lambda: konrad.lapserate.MoistLapseRate()


## 8. Convective adjustment
### Default (Strict adjustment towards the convectively-stable temperature profile)
hard = lambda: konrad.convection.HardAdjustment(etol=etol)


## 9. Ozone model
### Default (Fixed ozone in pressure levels)
fixdO3p = lambda: konrad.ozone.OzonePressure()
### Cariolle chemistry (linearised)
dynamO3 = lambda upc: konrad.ozone.Cariolle(is_coupled_upwelling=upc)


## 10. Surface model (default is SlabOcean)
### Default (Fixed surface temperature with default settings)
def fixdT(alpha=0.2, T=288.0, epsilon=1.0, z=0.0):
    return konrad.surface.FixedTemperature(
        albedo=alpha, temperature=T, longwave_emissivity=epsilon, height=z
    )

### Slab ocean with thermal capacity and heat transport
def SlabO(alpha=0.2, T=288.0, epsilon=1.0, z=0.0, d=1.0, hs=66.0):
    return konrad.surface.SlabOcean(
        albedo=alpha,
        temperature=T,
        longwave_emissivity=epsilon,
        height=z,
        depth=d,
        heat_sink=hs,
    )

### Fixed surface temperature from file
def fixdT_from_file(file, ts=-1):
    """Defines a `FixedTemperature` surface model from a file.
        Useful for creating restarts
    """
    # Imports loader of netCDFs
    from netCDF4 import Dataset as ncload

    # Initialises boolean variable
    use_ff = False
    # Checks if the file has surface model of type `FixedTemperature`
    with ncload(file, mode="r") as temp:
        if temp.groups["surface"].getncattr("class") == "FixedTemperature":
            use_ff = True
    # If not, it directly extracts the data and defines the new surface model
    if not use_ff:
        with ncload(file, mode="r") as temp:
            Ts = (
                temp.groups["surface"]
                .variables["temperature"][ts]
                .data.astype("float64")
            )
            h = float(temp.groups["surface"].variables["height"][:])
            alpha = float(temp.groups["surface"].variables["albedo"][:])
            epsilon = float(temp.groups["surface"].variables["longwave_emissivity"][:])
            return fixdT(alpha=alpha, T=Ts, epsilon=epsilon, z=h)
    # If the in-file surface model is of type `FixedTemperature`, then simply copies it
    else:
        return konrad.surface.FixedTemperature().from_netcdf(file, timestep=ts)

### Slab ocean from file
def SlabO_from_file(file, depth=1.0, ts=-1):
    """Defines a `SlabOcean` surface model from a file. Useful for creating restarts"""
    # Imports loader of netCDFs
    from netCDF4 import Dataset as ncload

    # Initialises boolean variable
    use_ff = False
    # Checks if the file has surface model of type `SlabOcean`
    with ncload(file, mode="r") as temp:
        if temp.groups["surface"].getncattr("class") == "SlabOcean":
            use_ff = True
    # If not, it directly extracts the data and defines the new surface model
    if not use_ff:
        with ncload(file, mode="r") as temp:
            Ts = (
                temp.groups["surface"]
                .variables["temperature"][ts]
                .data.astype("float64")
            )
            h = float(temp.groups["surface"].variables["height"][:])
            alpha = float(temp.groups["surface"].variables["albedo"][:])
            epsilon = float(temp.groups["surface"].variables["longwave_emissivity"][:])
            d = depth
            hs = float(temp.groups["radiation"].variables["toa"][ts]) - correction
            return SlabO(alpha=alpha, T=Ts, epsilon=epsilon, z=h, d=d, hs=hs)
    # If the in-file surface model is of type `SlabOcean`, then simply copies it
    else:
        return konrad.surface.SlabOcean().from_netcdf(file, timestep=ts)


## 11. Cloud model
### Default (clear sky)
clear_sky = lambda: konrad.cloud.ClearSky(nlev)


## 12. Stratospheric upwelling model
### Stratospheric upwelling with a constant vertical velocity
var_up = lambda w: konrad.upwelling.StratosphericUpwelling(w=w)
### Stratospheric upwelling linked to surface warming
def upT(w=0.2, rate=0.03):
    return konrad.upwelling.Coupled_StratosphericUpwelling(
        w=w, rate=rate
    )


## 13. Diurnal cycle switch (Zenithal angle variation)
diurnal_cycle = False  # default


## 14. Convergence, timing and output settings
### Convergence threshold (Radiative flux at the TOA)
delta = 0.0
delta2 = 0.0
### Time interval for writing output
writeevery = "120h"
### Iteration interval for on-screen logging
logevery = 120
### Numerical model time step
timestep = datetime.timedelta(seconds=timestep) # for normal runs seconds=1
### Maximum duration of the experiment in days
max_duration = datetime.timedelta(days=maxdur) # 7300 before
### Time that it should be in quasiequilibrium conditions
post_count = "365d" # 730 before


## 15. Time-step adjuster
### Increment for the adjustment
incr = datetime.timedelta(seconds=incr) # for normal runs seconds=900, otherwise 60
### Minimum time step
tmin = datetime.timedelta(seconds=tmin) # for normal runs seconds=1
### Maximum time step
tmax = datetime.timedelta(seconds=tmax) # for normal runs seconds=7200, otherwise 450
### Temperature difference threshold for lenghtening the time step
dlow = 0.001
### Temperature difference threshold for shortening the time step
dupp = 0.1
### Time-step adjuster object
tmstp_adj = lambda inc, tmi, tma, lo, up: konrad.core.TimestepAdjuster(
    increment=inc, timestep_min=tmi, timestep_max=tma, lower=lo, upper=up
)


# conv-O3 experiment definitions
## Constructs the dictionary with the experiment definition, putting default values
experiments = [base_name]
runs={}
for exp in experiments:
    runs[exp] = {}
    runs[exp]["run"] = {
        "atmosphere": None,
        "radiation": radt(),
        "humidity": def_h(UTRH),
        "lapserate": moist(),
        "convection": hard(),
        "ozone": dynamO3(True),
        "surface": None,
        "cloud": clear_sky(),
        "upwelling": var_up(upw),
        "diurnal_cycle": diurnal_cycle,
        "delta": delta,
        "delta2": delta2,
        "writeevery": writeevery,
        "logevery": logevery,
        "timestep": timestep,
        "max_duration": max_duration,
        "post_count": post_count,
        "timestep_adjuster": tmstp_adj(incr, tmin, tmax, dlow, dupp),
        "experiment": exp,
        "outfile": output_dir + "_".join(exp.split(" ")) + ".nc",
    }
    runs[exp]["restart"] = {
        "irst": False,
        "rst_file": None,
    }
    runs[exp]["nxCO2"] = CO2


# Define the relative humidity function in case it is not the default constant.
if RHF == 1:
    runs[exp]["run"]["humidity"] = csh_h(UTRH)
elif RHF == 2:
    runs[exp]["run"]["humidity"] = csp_h(UTRH)
elif RHF == 3:
    runs[exp]["run"]["humidity"] = ERA_h()


# Define the upwelling function in case it is evolving.
if upw == 1.0:
    runs[exp]["run"]["upwelling"] = upT(w=0.2, rate=0.01)
if upw == 2.0:
    runs[exp]["run"]["upwelling"] = upT(w=0.2, rate=0.02)
if upw == 3.0:
    runs[exp]["run"]["upwelling"] = upT(w=0.2, rate=0.03)
if upw == 4.0:
    runs[exp]["run"]["upwelling"] = upT(w=0.2, rate=0.04)
if upw == 5.0:
    runs[exp]["run"]["upwelling"] = upT(w=0.2, rate=0.05)
if upw == 6.0:
    runs[exp]["run"]["upwelling"] = upT(w=0.2, rate=0.06)
if upw == 7.0:
    runs[exp]["run"]["upwelling"] = upT(w=0.2, rate=0.07)
if upw == 8.0:
    runs[exp]["run"]["upwelling"] = upT(w=0.2, rate=0.08)
if upw == 9.0:
    runs[exp]["run"]["upwelling"] = upT(w=0.2, rate=0.09)
if upw == 10.0:
    runs[exp]["run"]["upwelling"] = upT(w=0.2, rate=0.10)
if upw == 11.0:
    if O3 == "f":
        runs[exp]["run"]["upwelling"] = var_up(0.30713043213)
    elif O3 == "a":
        runs[exp]["run"]["upwelling"] = var_up(0.29812164307)


# Particular configuration for runs that are not Fixed-SST controls
if not is_cntrl:
    runs[exp]["restart"]["irst"] = True
    runs[exp]["restart"]["rst_file"] = output_dir + "_".join(cntrl_nam.split(" ")) + ".nc"
    runs[exp]["run"]["surface"] = SlabO()

# Configuration for all runs
if O3 == "f": # Fixed ozone
    runs[exp]["run"]["ozone"] = fixdO3p()

# Initialisation functions
def ini_atmo(run_conf):
    """Initialises an `Atmosphere` model using a run configuration dictionary"""
    # If this is a restart, it generates the new atmosphere from file
    if run_conf["restart"]["irst"]:
        print("Initialising atmosphere model with atmosphere restart data")
        return atmo().from_netcdf(run_conf["restart"]["rst_file"], timestep=-1)
    # In any other case, it uses the definitions from the dictionary
    else:
        atm = atmo()
        atm["T"] = np.array([CARIOLLE_Tp(plev)])
        atm["O3"] = np.array([CARIOLLE_O3p(plev)])
        print("Initialising atmosphere model with Cariolle temperature profile")
        return atm

def ini_surf(run_conf, atm):
    """Initialises a `Surface` model using a run configuration dictionary"""
    # If this is not a restart, it uses a default `FixedTemperature` model
    if not run_conf["restart"]["irst"]:
        temp = fixdT(T=295.0)
        return temp
        print("Initialising surface model with atmosphere and default data")
    # In any other case, it restarts according to the dictionary's surface type
    else:
        print("Initialising surface model with surface restart data")
        if type(run_conf["run"]["surface"]) == konrad.surface.FixedTemperature:
            return fixdT_from_file(run_conf["restart"]["rst_file"])
        if type(run_conf["run"]["surface"]) == konrad.surface.SlabOcean:
            return SlabO_from_file(run_conf["restart"]["rst_file"])


# Increment function
def nxCO2(atm, n=1.0):
    atm["CO2"] *= n
    print("Multiplying CO2 concentration by {}".format(n))


# Main cycle
runs_continue = experiments

for actual in runs_continue:
    # Initialises atmosphere
    runs[actual]["run"]["atmosphere"] = ini_atmo(runs[actual])
    # Initialises surface
    runs[actual]["run"]["surface"] = ini_surf(
        runs[actual], runs[actual]["run"]["atmosphere"]
    )
    # Multiplies the atmospheric carbon dioxide concentration
    nxCO2(runs[actual]["run"]["atmosphere"], n=runs[actual]["nxCO2"])
    # Enables on-screen log
    konrad.enable_logging()
    # Initialises the numerical model
    rce = konrad.RCE(**runs[actual]["run"])
    # Run it except there are exceptions, but continues with the next experiment
    try:
        rce.run()
    except Exception as e:
        traceback.print_exc()
