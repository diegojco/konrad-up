""" Module containing convenience functions for konrad experiments of the
    upwelling-ozone project
"""
""" External Modules
"""

import calctool as clima
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LogFormatter

""" Matplotlib customised tick formatters
"""

class HectoPascalLogFormatter(LogFormatter):
    def _num_to_string(self, x, vmin, vmax):
        return '{:g}'.format(x / 100)
class normalFormatter(LogFormatter):
    def _num_to_string(self, x, vmin, vmax):
        return '{:g}'.format(x)

""" Constants
"""

cst = {
    "base": "/home/mpim/m300556/konrad_exps/output_files",
    "prefix": "conv-O3_last-",
    "exps": {
        "control": {
            "suffix": "_cntrl",
            "CO2": 1.0,
        },
        "pi-control": {
            "suffix": "_picntrl",
            "CO2": 1.0,
        },
        "upw-oz": {
            "suffix": "",
            "CO2": 1.0,
        },
        "2xCO2": {
            "suffix": "",
            "CO2": 2.0,
        },
        "2xCO2*": {
            "suffix": "_alt_1",
            "CO2": 2.0,
        },
    },
    "cntrls": ["control", "pi-control"],
    "no_cntrls": ["upw-oz", "2xCO2", "2xCO2*"],
    "RHs": [0.4, 0.7, 2.8, 3.6],
    "RHs_s": {
        "U4": 0.4,
        "CB": 3.6,
        "U7": 0.7,
        "C8": 2.8,
    },
    "RHs_v": {
        0.4: "U4",
        3.6: "CB",
        0.7: "U7",
        2.8: "C8",
    },
    "RHs_c": {
        0.4: tuple(clima.np.array([94, 60, 153]) / 255),
        3.6: tuple(clima.np.array([178, 171, 210]) / 255),
        0.7: tuple(clima.np.array([230, 97, 1]) / 255),
        2.8: tuple(clima.np.array([253, 184, 99]) / 255),
    },
    "O3s": ["f", "fh", "a", "b"],
    "O3s_l": {
        "f": "O$_{3}$ prescribed (p)",
        "fh": "O$_{3}$ prescribed (z)",
        "a": "Cariolle O$_{3}$",
        "b": "Cariolle alt O$_{3}$",
    },
    "ws": [
        0.000, 0.025, 0.050, 0.075,
        0.100, 0.125, 0.150, 0.175,
        0.200, 0.225, 0.250, 0.275,
        0.300, 0.325, 0.350, 0.375,
        0.400,
    ],
    "m_val": {
        "p01": 0.01,
        "p02": 0.02,
        "p03": 0.03,
        "p04": 0.04,
        "p05": 0.05,
        "p06": 0.06,
        "p07": 0.07,
        "p08": 0.08,
        "p09": 0.09,
        "p10": 0.10,
    },
    "ws_var_val": {
        "p01": {
            "f": 0.227,
            "a": 0.225,
        },
        "p02": {
            "f": 0.250,
            "a": 0.247,
        },
        "p03": {
            "f": 0.271,
            "a": 0.266,
        },
        "p04": {
            "f": 0.290,
            "a": 0.283,
        },
        "p05": {
            "f": 0.307,
            "a": 0.298,
        },
        "p06": {
            "f": 0.323,
            "a": 0.312,
        },
        "p07": {
            "f": 0.338,
            "a": 0.325,
        },
        "p08": {
            "f": 0.351,
            "a": 0.336,
        },
        "p09": {
            "f": 0.364,
            "a": 0.347,
        },
        "p10": {
            "f": 0.375,
            "a": 0.357,
        },
        "p11": {
            "f": 0.307,
            "a": 0.298,
        },
    },
    "ws_cs": {
        0.000: 0.000 * 2.0,
        0.025: 0.025 * 2.0,
        0.050: 0.050 * 2.0,
        0.075: 0.075 * 2.0,
        0.100: 0.100 * 2.0,
        0.125: 0.125 * 2.0,
        0.150: 0.150 * 2.0,
        0.175: 0.175 * 2.0,
        0.200: 0.200 * 2.0,
        0.225: 0.225 * 2.0,
        0.250: 0.250 * 2.0,
        0.275: 0.275 * 2.0,
        0.300: 0.300 * 2.0,
        0.325: 0.325 * 2.0,
        0.350: 0.350 * 2.0,
        0.375: 0.375 * 2.0,
        0.400: 0.400 * 2.0,
        "p01": 0.227 * 2.0,
        "p02": 0.250 * 2.0,
        "p03": 0.271 * 2.0,
        "p04": 0.290 * 2.0,
        "p05": 0.307 * 2.0,
        "p06": 0.323 * 2.0,
        "p07": 0.338 * 2.0,
        "p08": 0.351 * 2.0,
        "p09": 0.364 * 2.0,
        "p10": 0.375 * 2.0,
        "p11": 0.307 * 2.0,
    },
    "vars": {
        "t": {
            "group": "",
            "name": "time",
        },
        "p": {
            "group": "",
            "name": "plev",
        },
        "ph": {
            "group": "",
            "name": "phlev",
        },
        "z": {
            "group": "atmosphere",
            "name": "z",
        },
        "T": {
            "group": "atmosphere",
            "name": "T",
        },
        "rv": {
            "group": "atmosphere",
            "name": "H2O",
        },
        "rv_O3": {
            "group": "atmosphere",
            "name": "O3",
        },
        "φ_t": {
            "group": "radiation",
            "name": "toa",
        },
        "φ_sw↑": {
            "group": "radiation",
            "name": "sw_flxu",
        },
        "φ_sw↓": {
            "group": "radiation",
            "name": "sw_flxd",
        },
        "φ_lw↑": {
            "group": "radiation",
            "name": "lw_flxu",
        },
        "φ_lw↓": {
            "group": "radiation",
            "name": "lw_flxd",
        },
        "Q_rad_sw": {
            "group": "radiation",
            "name": "sw_htngrt",
        },
        "Q_rad_lw": {
            "group": "radiation",
            "name": "lw_htngrt",
        },
        "Q_rad": {
            "group": "radiation",
            "name": "net_htngrt",
        },
        "Q_up": {
            "group": "upwelling",
            "name": "cooling_rates",
        },
        "Q_conv": {
            "group": "convection",
            "name": "convective_heating_rate",
        },
        "ctop": {
            "group": "convection",
            "name": "convective_top_index",
        },
        "T_s": {
            "group": "surface",
            "name": "temperature",
        },
        "H": {
            "group": "surface",
            "name": "heat_sink",
        },
    },
    "dvars": {
        "φ_sw": {
            "function": clima.φ_sw,
            "args": ("φ_sw↑", "φ_sw↓" ),
        },
        "φ_lw": {
            "function": clima.φ_lw,
            "args": ("φ_lw↑", "φ_lw↓" ),
        },
        "φ_↑": {
            "function": clima.φ_u,
            "args": ("φ_sw↑", "φ_lw↑" ),
        },
        "φ_↓": {
            "function": clima.φ_d,
            "args": ("φ_sw↓", "φ_lw↓" ),
        },
        "φ": {
            "function": clima.φ,
            "args": ("φ_sw↑", "φ_lw↑", "φ_sw↓", "φ_lw↓" ),
        },
        "φ_s": {
            "function": clima.φ_s,
            "args": ("φ_sw↑", "φ_lw↑", "φ_sw↓", "φ_lw↓" ),
        },
        "q": {
            "function": clima.WV_rv_to_q,
            "args": ("rv", ),
        },
        "q_O3": {
            "function": clima.O3_rv_to_q,
            "args": ("rv_O3", ),
        },
        "RH": {
            "function": clima.WV_RH,
            "args": ("rv", "T", "p", ),
        },
        "cp": {
            "function": clima.cold_point_tropopause_index_calc,
            "args": ("p", "T", "z", ),
        },
        "Γ": {
            "function": clima.Γ_calc,
            "args": ("T", "z", ),
        },
        "ΔΓ": {
            "function": clima.ΔΓ_calc,
            "args": ("T", "z", ),
        },
        "Γ_m": {
            "function": clima.Γ_m_calc,
            "args": ("T", "p", ),
        },
    },
    "csvars": [
        "ΔT_s", "N", "S", "λ", "F_eff", "F_eff,f", "N_f", "φ_t,f", "T_s,f"
    ],
    "maxp": 5000.0,
    "minp": 10000.0,
}

""" Create variable vs w diagrams
"""
def diagrams_vw(
    variables=["N_f"],
    experiment="2xCO2",
    RHs=[0.4],
    O3s=["f"],
    ws=cst["ws"],
    filename="test.pdf",
    ylabels=["$N_{f}\,/\,\mathrm{W}\,\mathrm{m}^{-2}$"],
    limits=[(0, 4)],
    columns=1,
    save=True,
    close=True,
    labels=False,
    labelRH=False,
    labelsO3=[r"prescribed (p) O$_{3}$"],
    prefix=cst["prefix"],
    base=cst["base"],
):
    RHsv = []
    for RH in RHs:
        if isinstance(RH, str):
            RHsv.append(cst["RHs_s"][RH])
        else:
            RHsv.append(RH)
    data = create_basic_data_repo(
        vs=["T_s", "φ_t"], es=["pi-control"] + [experiment], RHs=RHsv, O3s=O3s, ws=ws,
        prefix=prefix, base=base,
    )
    csdata = create_FFCS_data_repo(
        data, es=[experiment], RHs=RHsv, O3s=O3s, ws=ws,
    )
    data = csdata
    if close:
        figures_vw(
            data, variables, experiment, RHsv, O3s, ws, filename, ylabels, limits,
            columns, save, close, labels, labelRH, labelsO3,
            prefix, base,
        )
        return data
    else:
        fig, axes = figures_vw(
            data, variables, experiment, RHsv, O3s, ws, filename, ylabels, limits,
            columns, save, close, labels, labelRH, labelsO3,
            prefix, base,
        )
        return fig, axes, data

def figures_vw(
    data,
    variables,
    experiment,
    RHs,
    O3s,
    ws,
    filename,
    ylabels,
    limits,
    columns,
    save,
    close,
    labels,
    labelRH,
    labelsO3,
    prefix,
    base,
):
    n = len(variables)
    rows = (n // columns)
    if n % columns != 0:
        rows += 1
    if n < columns:
        columns = n
        rows = 1
    bottom = scan_bottom(n, rows, columns)
    bottom = [variables[i] for i in bottom]
        
    fig = plt.figure(figsize=(columns * 3.25, rows * 3.25), dpi=400)
    axes = {variables[i]: fig.add_subplot(rows, columns, i + 1) for i in range(n)}
    limitsd = {variables[i]: limits[i] for i in range(n)}
    ylabelsd = {variables[i]: ylabels[i] for i in range(n)}
    labelsO3d = {O3s[i]: labelsO3[i] for i in range(len(O3s))}
    for variable in axes:
        if variable == "N_f":
            axes[variable].axhline(0, lw=0.7, ls="-", c=(0,0,0))
            axes[variable].axvline(0.2, lw=0.7, ls="-", c=(0,0,0))
        for RH in RHs:
            tempc = cst["RHs_c"][RH]
            templabel = ""
            if labelRH:
                templabel += f"{cst['RHs_v'][RH]} "
            for O3 in O3s:
                if ((len(O3s) == 1) or (O3 == "f")):
                    templs = "-"
                else:
                    templs = "--"
                tempv = [data[variable][experiment][RH][O3][w] for w in ws]
                tempws = []
                for w in ws:
                    if isinstance(w, float):
                        tempws.append(w)
                    else:
                        we = upwelling_speed_from_text(w, RH, O3, prefix, base) 
                        tempws.append(we)
                tempwsv = {tempws[i]: tempv[i] for i in range(len(tempws))}
                tempws.sort()
                tempv = [tempwsv[w] for w in tempws]
                axes[variable].plot(
                    tempws, tempv, lw=1.8, ls=templs, c=tempc,
                    label=templabel + labelsO3d[O3]
                )
                del tempwsv
        if variable in bottom:
            axes[variable].set_xlabel(
                "$w\,/\,\mathrm{mm}\,\mathrm{s}^{-1}$", fontsize=7
            )
        else:
            axes[variable].set_xticklabels([])
        axes[variable].set_ylim(*limitsd[variable])
        axes[variable].set_xlim((-0.02,0.42))
        axes[variable].locator_params(axis="x", nbins=5)
        axes[variable].tick_params(which="both", labelsize=7)
        axes[variable].set_ylabel(ylabelsd[variable], fontsize=7)
        if labels and (variable == variables[0]):
            axes[variable].legend(frameon=False, fancybox=False, fontsize=5)
    fig.tight_layout()
    if save:
        fig.savefig(filename, bbox_inches="tight")
    if close:
        plt.close(fig)
    else:
        return fig, axes

""" Create NT-diagrams
"""
def diagram_NT(
    e="2xCO2",
    RH=0.4,
    O3="f",
    ws=cst["ws"],
    fname="test.pdf",
    lims=(0, 4),
    save=True,
    close=True,
    picontrol=True,
    labels=False,
    prefix=cst["prefix"],
    base=cst["base"],
):
    RHv = None
    if isinstance(RH, str):
        RHv = cst["RHs_s"][RH]
    else:
        RHv = RH
    data = create_basic_data_repo(
        vs=["T_s", "φ_t"], es=["pi-control"] + [e], RHs=[RHv], O3s=[O3], ws=ws,
        prefix=prefix, base=base,
    )
    csdata = create_FFCS_data_repo(
        data, es=[e], RHs=[RHv], O3s=[O3], ws=ws,
    )
    data = csdata
    if close:
        figure_NT(
            data, e, RHv, O3, ws, fname, lims,
            save, close, picontrol, labels,
            prefix, base,
        )
        return data
    else:
        fig, axes = figure_NT(
            data, e, RHv, O3, ws, fname, lims,
            save, close, picontrol, labels,
            prefix, base,
        )
        return fig, axes, data

def figure_NT(
    data,
    e,
    RH,
    O3,
    ws,
    fname,
    lims,
    save,
    close,
    picontrol,
    labels,
    prefix,
    base,
):
    fig = plt.figure(figsize=(3.25, 3.25), dpi=400)
    axes = fig.add_subplot(1,1,1)
    axes.axhline(0, lw=0.7, ls="-", c=(0,0,0))
    axes.axvline(0, lw=0.7, ls="-", c=(0,0,0))
    for w in ws:
        tempc = colorscale(cst["RHs_c"][RH], cst["ws_cs"][w])
        if len(ws) == 1:
            tempc = cst["RHs_c"][RH]
        tempT = data["ΔT_s"][e][RH][O3][w]
        tempTt = clima.np.linspace(0, data["ΔT_s"][e][RH][O3][w].max())
        tempN0 = -data["N_f"][e][RH][O3][w]
        tempN = data["N"][e][RH][O3][w]
        tempNt = (data["λ"][e][RH][O3][w] * tempTt) + data["F_eff,f"][e][RH][O3][w]
        if picontrol:
            tempN0 = tempN0 + data["N_f"][e][RH][O3][w]
            tempN = tempN + data["N_f"][e][RH][O3][w]
            tempNt = tempNt + data["N_f"][e][RH][O3][w]
        axes.plot(
            tempTt, tempNt,
            lw=1.8, ls="--", c=(0.5,0.5,0.5)
        )
        axes.plot(
            0, tempN0,
            ls="", marker="o", markersize=2, c=tempc,
        )
        if isinstance(w, float):
            axes.plot(
                tempT, tempN,
                ls="", marker="o", markersize=2, c=tempc,
                label=f"$w={w:0.3f}" + "\,\mathrm{mm}\,\mathrm{s}^{-1}$"
            )
        else:
            if w == "p11":
                if O3 == "f":
                    we = 0.307
                else:
                    we = 0.298
            else:
                we = upwelling_speed_from_text(w, RH, O3, prefix, base)
            axes.plot(
                tempT, tempN,
                ls="", marker="o", markersize=2, c=tempc,
                label=(
                    f"$w_{{e}}={we:0.3f}" +
                    "\,\mathrm{mm}\,\mathrm{s}^{-1}$"
                )
            )
    axes.set_xlim(*lims)
    axes.locator_params(axis="x", nbins=5)
    axes.tick_params(which="both", labelsize=7)
    axes.set_xlabel("$\Delta T_{\mathrm{s}}\,/\,\mathrm{K}$", fontsize=7)
    axes.set_ylabel("$N\,/\,\mathrm{W}\,\mathrm{m}^{-2}$", fontsize=7)
    if labels:
        axes.legend(frameon=False, fancybox=False, fontsize=5)
    fig.tight_layout()
    if save:
        fig.savefig(fname, bbox_inches="tight")
    if close:
        plt.close(fig)
    else:
        return fig, axes

""" Create 2xCO2 profile figures
"""

def profiles_doubling(
    variables={"T": {0.4: {"f": [0.2]}}},
    experiment="2xCO2",
    filename="test.pdf",
    xlabels={"T": "$T\,/\,\mathrm{K}$"},
    limits={"T": (185, 305)},
    columns=1,
    save=False,
    close=False,
    labels=[True],
    labelRH=False,
    labelsO3={"f": r"prescribed (p) O$_{3}$"},
    ifleft=False,
    prefix=cst["prefix"],
    base=cst["base"],
):
    b_vars, d_vars, RHs, O3s, w_set = _extract_profile_plotting_info(variables)
    
    data = create_basic_data_repo(
        vs=b_vars, es=["pi-control", experiment],
        RHs=RHs, O3s=O3s, ws=w_set,
        prefix=prefix, base=base,
    )
    if len(d_vars) != 0:
        derived = create_derived_data_repo(
            data, dvs=d_vars, es=["pi-control", experiment],
            RHs=RHs, O3s=O3s, ws=w_set,
        )
        data = {**data, **derived}
        del derived
        
    if close:
        figures_doubling(
            data, variables, filename,
            xlabels, limits, columns, save, close, labels, labelRH, labelsO3, ifleft,
            prefix, base,
        )
        return data
    else:
        fig, axes = figures_doubling(
            data, variables, filename,
            xlabels, limits, columns, save, close, labels, labelRH, labelsO3, ifleft,
            prefix, base
        )
        return fig, axes, data

def figures_doubling(
    data, variables, filename,
    xlabels, limits, columns, save, close, labels, labelRH, labelsO3, ifleft,
    prefix, base,
):
    
    triplets = _parameter_triplets(variables)
    n, rows, cols = _figure_array_info(triplets, columns)
    left = scan_left(n, rows, cols)
    left = [triplets[i] for i in left]
    
    fig, axes = _figure_array_construct(n, rows, cols, triplets)
    xlabelsd = {triplets[i]: xlabels[triplets[i][0]] for i in range(n)}
    scaled = {triplets[i]: _scale_factors(triplets[i][0]) for i in range(n)}
    maskd = {triplets[i]: _vertical_masks(triplets[i][0], data["p"]) for i in range(n)}
    limitsd = {triplets[i]: limits[triplets[i][0]] for i in range(n)}
    labelsO3d = {triplets[i]: labelsO3[triplets[i][2]] for i in range(n)}
    labelsd = {triplets[i]: labels[i] for i in range(n)}
    
    for t in triplets:
        tempc = cst["RHs_c"][t[1]]
        templabel = ""
        if labelRH:
            templabel += f"{cst['RHs_v'][t[1]]} "
        templabel += labelsO3d[t]
        
        axes[t].plot(
            data[t[0]]["pi-control"][t[1]][-20:,...].mean(axis=0)[maskd[t]] * scaled[t],
            data["p"][maskd[t]],
            label=templabel + "$\mathtt{control}$",
            lw=1.8, ls="--", c=cst["RHs_c"][t[1]]
        )
        
        for w in variables[t[0]][t[1]][t[2]]:
            temp = data[t[0]]["2xCO2"][t[1]][t[2]][w][-20:,...].mean(axis=0)[maskd[t]]
            if isinstance(w, float):
                axes[t].plot(
                    (temp * scaled[t]),
                    data["p"][maskd[t]],
                    label=(
                        templabel +
                        f" $w={w:0.3f}" + "\,\mathrm{mm}\,\mathrm{s}^{-1}$"
                    ),
                    lw=1.8, ls="-", c=colorscale(tempc, w * 2.0)
                )
            else:
                we = upwelling_speed_from_text(w, t[1], t[2], prefix, base)
                axes[t].plot(
                    (temp * scaled[t]),
                    data["p"][maskd[t]],
                    label=(
                        templabel +
                        f" $w_{{e}}={we:0.3f}" +
                        "\,\mathrm{mm}\,\mathrm{s}^{-1}$"
                    ),
                    lw=1.8, ls="-", c=colorscale(tempc, we * 2.0)
                )
        axes[t].set_xlim(*limitsd[t])
        axes[t].locator_params(axis="x", nbins=5)
        axes[t].tick_params(which="both", labelsize=7)
        axes[t].set_xlabel(xlabelsd[t], fontsize=7)
        axes[t].invert_yaxis()
        axes[t].set_yscale("log")
        if ifleft:
            if t in left:
                axes[t].set_ylabel("$p\,/\,\mathrm{hPa}$", fontsize=7)
                axes[t].yaxis.set_minor_formatter(HectoPascalLogFormatter())
                axes[t].yaxis.set_major_formatter(HectoPascalLogFormatter())
            else:
                axes[t].set_yticklabels([])
                axes[t].set_yticklabels([], minor=True)
        else:
            axes[t].set_ylabel("$p\,/\,\mathrm{hPa}$", fontsize=7)
            axes[t].yaxis.set_minor_formatter(HectoPascalLogFormatter())
            axes[t].yaxis.set_major_formatter(HectoPascalLogFormatter())
        if labelsd[t]:
            axes[t].legend(frameon=False, fancybox=False, fontsize=5)
        if t[0] == "q":
            axes[t].set_xscale("log")
    
    fig.tight_layout()
    if save:
        fig.savefig(filename, bbox_inches="tight")
    if close:
        plt.close(fig)
    else:
        return fig, axes

def _extract_profile_plotting_info(variables):
    b_vars = set()
    d_vars = set()
    n_vars = set()
    RHs = set()
    O3s = set()
    w_set = set()
    for variable in variables:
        if variable in cst["vars"]:
            b_vars = b_vars.union([variable])
        elif variable in cst["dvars"]:
            d_vars = d_vars.union([variable])
            n_vars = n_vars.union(list(cst["dvars"][variable]["args"]))
        
        for RH in variables[variable]:
            if isinstance(RH, str):
                RHs = RHs.union([cst["RHs_s"][RH]])
            else:
                RHs = RHs.union([RH])
            
            for O3 in variables[variable][RH]:
                O3s = O3s.union([O3])
                
                w_set = w_set.union(set(variables[variable][RH][O3]))
    
    b_vars = b_vars.union(n_vars)
    b_vars = ["p"] + list(b_vars)
    d_vars = list(d_vars)
    RHs = list(RHs)
    O3s = list(O3s)
    w_set = list(w_set)
    return b_vars, d_vars, RHs, O3s, w_set

def _figure_array_construct(n, rows, cols, triplets):
    fig = plt.figure(figsize=(cols * 3.25 , rows * 3.25), dpi=400)
    axes = {
        triplets[i]: fig.add_subplot(rows, cols, i + 1) for i in range(n)
    }
    return fig, axes

def _figure_array_info(triplets, columns):
    cols = columns
    n = len(triplets)
    rows = (n // cols)
    if n % cols != 0:
        rows += 1
    if n < cols:
        cols = n
        rows = 1
    return n, rows, cols

def _parameter_triplets(variables):
    triplets = []
    for variable in variables:
        for RH in variables[variable]:
            for O3 in variables[variable][RH]:
                triplets.append((variable, RH, O3))
    return triplets

def _scale_factors(variable):
    if variable in ["rv_O3"]:
        scale = 1e6
    elif variable in ["q_O3"]:
        scale = 1e6
    elif variable in ["q"]:
        scale = 1e3
    elif variable in ["Γ", "Γ_m", "Γ_d", "ΔΓ"]:
        scale = 1e3
    elif variable in ["RH"]:
        scale = 1.0
    else:
        scale = 1.0
    return scale

def _vertical_masks(variable, pressure):
    if variable in ["rv_O3"]:
        mask = maskminmax(pressure, limmin=10000.0, limmax=2000.0)
    elif variable in ["q_O3"]:
        mask = maskminmax(pressure, limmin=10000.0, limmax=2000.0)
    elif variable in ["Γ", "Γ_m", "Γ_d", "ΔΓ"]:
        mask = maskminmax(pressure, limmin=30000.0)
    elif variable in ["RH"]:
        mask = maskminmax(pressure, limmin=30000.0)
    else:
        mask = maskminmax(pressure, limmin=30000.0)
    return mask

""" Create control profile figures
"""

def profile_control(
    v="T",
    RHs=[0.4, 3.6],
    fname="test.pdf",
    xlabel="$T\,/\,\mathrm{K}$",
    lims=(185, 305),
    save=True,
    close=True,
    prefix=cst["prefix"],
    base=cst["base"],
):
    RHsv = []
    for RH in RHs:
        if isinstance(RH, str):
            RHsv.append(cst["RHs_s"][RH])
        else:
            RHsv.append(RH)
    if v in cst["vars"]:
        data = create_basic_data_repo(
            vs=["p", v], es=["control", "pi-control"], RHs=RHsv,
            prefix=prefix, base=base,
        )
    elif v in cst["dvars"]:
        otras = list(cst["dvars"][v]["args"])
        basic = create_basic_data_repo(
            vs=["p",] + otras, es=["control", "pi-control"], RHs=RHsv,
            prefix=prefix, base=base,
        )
        data = create_derived_data_repo(
            basic, dvs=[v], es=["control", "pi-control"], RHs=RHsv,
        )
        data = {**basic, **data}
    if close:
        figure_control(
            data, v, RHsv, fname, xlabel, lims, save=save, close=close
        )
    else:
        fig, axes = figure_control(
            data, v, RHsv, fname, xlabel, lims, save=save, close=close
        )
        return fig, axes

def figure_control(
    data,
    v,
    RHs,
    fname,
    xlabel,
    lims,
    save=True,
    close=True,
):
    if v == "q_O3":
        mask = maskminmax(data["p"], limmin=20000.0)
        scale = 1e6
    elif v in ["Γ", "Γ_m", "Γ_d", "ΔΓ"]:
        mask = maskminmax(data["p"], limmin=30000.0)
        scale = 1e3
    elif v in ["RH"]:
        mask = maskminmax(data["p"], limmin=100000.0)
        scale = 1.0
    else:
        mask = maskminmax(data["p"], limmin=30000.0)
        scale = 1.0
    
    fig = plt.figure(figsize=(3.25, 3.25), dpi=400)
    axes = fig.add_subplot(1,1,1)
    for RH in RHs:
        axes.plot(
            data[v]["control"][RH][-1,...][mask] * scale,
            data["p"][mask],
            label=f"{cst['RHs_v'][RH]}" + " $\mathtt{control}$",
            lw=1.8, ls="--", c=cst["RHs_c"][RH]
        )
        axes.plot(
            data[v]["pi-control"][RH][-1,...][mask] * scale,
            data["p"][mask],
            label=f"{cst['RHs_v'][RH]}" + " $\mathtt{pi-control}$",
            lw=1.8, ls="-", c=cst["RHs_c"][RH]
        )
    axes.set_xlim(*lims)
    axes.locator_params(axis="x", nbins=5)
    axes.tick_params(which="both", labelsize=7)
    axes.set_xlabel(xlabel, fontsize=7)
    axes.set_ylabel("$p\,/\,\mathrm{hPa}$", fontsize=7)
    axes.legend(frameon=False, fancybox=False, fontsize=5)
    axes.invert_yaxis()
    axes.set_yscale("log")
    if v == "q":
        axes.set_xscale("log")
    axes.yaxis.set_minor_formatter(HectoPascalLogFormatter())
    axes.yaxis.set_major_formatter(HectoPascalLogFormatter())
    fig.tight_layout()
    if save:
        fig.savefig(fname, bbox_inches="tight")
    if close:
        plt.close(fig)
    else:
        return fig, axes

def maskminmax(p, limmin=cst["minp"], limmax=cst["maxp"]):
    return clima.np.logical_and(maskmin(p, limmin=limmin), maskmax(p,limmax=limmax))

def maskmax(p, limmax=cst["maxp"]):
    return p > limmax

def maskmin(p, limmin=cst["minp"]):
    return p < limmin

def colorscale(color, x):
    return tuple(
        ((clima.np.array([1, 1, 1]) - clima.np.array(list(color))) * x) +
        clima.np.array(list(color))
    )

def scan_bottom(n, rows, columns):
    bottom = clima.np.array(range(n), dtype=int) + 1
    if n % columns != 0:
        bottom = clima.np.concatenate(
            (
                bottom,
                clima.np.zeros(columns - (n % columns), dtype=int)
            )
        )
    bottom = bottom.reshape((rows, columns))

    temp0 = bottom[-1][clima.np.where(bottom[-1] != 0)[0]]
    temp1 = clima.np.where(bottom[-1] == 0)[0]
    if len(temp1) != 0:
        bottom = clima.np.concatenate(
            (temp0, bottom[-2][temp1])
        ) - 1
    else:
        bottom = temp0 - 1
    return bottom

def scan_left(n, rows, columns):
    left = clima.np.array(range(n), dtype=int) + 1
    if n % columns != 0:
        left = clima.np.concatenate(
            (
                left,
                clima.np.zeros(columns - (n % columns), dtype=int)
            )
        )
    left = left.reshape((rows, columns))[:, 0] - 1

    return left

def upwelling_speed_from_text(wtext, RH, O3, prefix, base):
    m = cst["m_val"][wtext]
    data = create_basic_data_repo(
        vs=["T_s", "φ_t"], es=["pi-control", "2xCO2"],
        RHs=[RH], O3s=["f", "a"], ws=[wtext],
        prefix=prefix,
        base=base,
    )
    ΔTs = data["T_s"]["2xCO2"][RH][O3][wtext][-1] - data["T_s"]["pi-control"][RH][-1]
    return 0.2 + (m * ΔTs)

""" Create tables
"""

def table_upwelling_we(
    RH=0.4,
    ws=["p01", "p02", "p03", "p04", "p05", "p06", "p07", "p08", "p09", "p10",],
    prefix=cst["prefix"],
    base=cst["base"],
):
    we = {
        "f": None,
        "a": None
    }
    for O3 in we:
        we[O3] = {w: upwelling_speed_from_text(w, RH, O3, prefix, base) for w in ws}
    table = (
        "\\begin{tabular}{@{}S[table-format=1.2]S[table-format=1.3]" +
        "S[table-format=1.3]@{}}\n"
    )
    table += "    \\toprule\n"
    table += "    $m\,/\,\si{\mmsK}$ &\multicolumn{2}{c}{$w_{e}\,/\,\si{\mms}$}\\\\\n"
    table += "    \\cmidrule{2-3}\n"
    table += "     &prescribed O$_{3}$ &Cariolle O$_{3}$\\\\\n"
    table += "    \\midrule\n"
    line   = "    {0:4.2f} &{1:5.3f} &{2:5.3f}\\\\\n"
    
    for w in we["f"]:
        string = (cst['m_val'][w], we["f"][w], we["a"][w])
        table += line.format(*string)
    
    table += "    \\bottomrule\n"
    table += "\\end{tabular}"
    del we
    return table

def table_O3_comparison(
    experiment="2xCO2",
    RH=0.4,
    ws=[0.2, 0.3, 0.4],
    prefix=cst["prefix"],
    base=cst["base"],
):
    data = create_basic_data_repo(
        vs=["p", "rv_O3"], es=["pi-control", experiment],
        RHs=[RH], O3s=["a"], ws=ws,
        prefix=prefix, base=base,
    )
    
    plevs = {
        "30hPa": clima.np.where(data["p"] < 3100.0)[0][0],
        "50hPa": clima.np.where(data["p"] < 5100.0)[0][0]
    }
    
    rv_O3_CCMs = {
        "30hPa": {
            "pi-control": {
                "Dietmüller": 5.0,
                "Nowack": 4.5,
                "Marsh": 3.9,
            },
            "4xCO2":{
                "Dietmüller": 4.5,
                "Nowack": 3.9,
                "Marsh": 3.4,
            },
        },
        "50hPa": {
            "pi-control": {
                "Dietmüller": 1.5,
                "Nowack": 1.8,
                "Marsh": 1.3,
            },
            "4xCO2":{
                "Dietmüller": 0.9,
                "Nowack": 1.3,
                "Marsh": 1.0,
            },
        },
    }
    
    rv_O3_k = {
        plev: {
            "pi-control": 0.0,
            experiment: {
                w: 0.0 for w in ws
            },
        } for plev in plevs
    }
    
    for plev in rv_O3_k:
        for exp in rv_O3_k[plev]:
            if exp == "pi-control":
                rv_O3_k[plev][exp] = data["rv_O3"][exp][RH][-1, plevs[plev]] * 1e6
            else:
                for w in rv_O3_k[plev][exp]:
                    temp = data["rv_O3"][exp][RH]["a"][w][-1, plevs[plev]] * 1e6
                    rv_O3_k[plev][exp][w] = temp
    del data, plevs, temp
    
    table = table_O3(rv_O3_CCMs, rv_O3_k, RH, ws, experiment, prefix, base)
    del rv_O3_CCMs, rv_O3_k
    return table

def table_O3(rv_O3_CCMs, rv_O3_k, RH, ws, exp, prefix, base):
    table = (
        "\\begin{tabular}{@{}rS[table-format=1.2]S[table-format=1.2]c" +
        "S[table-format=1.2]S[table-format=1.2]c" +
        "S[table-format=1.2]S[table-format=1.2]@{}}\n"
    )
    table += "    \\toprule\n"
    table += (
        "      &\multicolumn{8}{c}{O$_{3}$ concentration$\,/\,\mathrm{ppmv}$}" +
        "\\\\\n"
    )
    table += "    \\cmidrule{2-9}\n"
    table += (
        "    \multirow{2}{*}{Model} &\multicolumn{2}{c}{\SI{30}{\hecto\Pa}} " +
        "& &\multicolumn{2}{c}{\SI{50}{\hecto\Pa}} & &\multicolumn{2}{c}{$\Delta$}" +
        "\\\\\n"
    )
    table += "     \\cmidrule{2-3}\\cmidrule{5-6}\\cmidrule{8-9}\n"
    table += (
        "      &\\texttt{nxCO2} &\\texttt{control} & &\\texttt{nxCO2} " +
        "&\\texttt{control} & &\SI{30}{\hecto\Pa} &\SI{50}{\hecto\Pa}\\\\\n"
    )
    table += "      \\midrule\n"
    line1  = (
        "      {0} &{1:4.2f} &{2:4.2f} & &{3:4.2f} &{4:4.2f} & &{5:4.2f} &{6:4.2f}" +
        "\\\\\n"
    )
    line2  = (
        "      {0} &{1:4.2f} & & &{3:4.2f} & & &{5:4.2f} &{6:4.2f}" +
        "\\\\\n"
    )
    for model in rv_O3_CCMs["30hPa"]["pi-control"]:
        string = (model, )
        for plev in rv_O3_CCMs:
            string = string + (
                rv_O3_CCMs[plev]["4xCO2"][model],
                rv_O3_CCMs[plev]["pi-control"][model],
            )
        for plev in rv_O3_CCMs:
            string = string + (
                rv_O3_CCMs[plev]["4xCO2"][model] -
                rv_O3_CCMs[plev]["pi-control"][model],
            )
        table += line1.format(*string)
    
    table += "    \\midrule\n"
    
    i=0
    for w in ws:
        if isinstance(w, float):
            string = ("\\texttt{"+ exp + "}" + f" $w = {w:5.3f}$",)
        else:
            we = upwelling_speed_from_text(w, RH, "a", prefix, base)
            string = ("\\texttt{"+ exp + "}" + f" $w_{{e}} = {we:5.3f}$",)
        for plev in rv_O3_k:
            string = string + (
                rv_O3_k[plev][exp][w],
                rv_O3_k[plev]["pi-control"],
            )
        for plev in rv_O3_k:
            string = string + (
                rv_O3_k[plev][exp][w] -
                rv_O3_k[plev]["pi-control"],
            )
        if i == 0:
            table += line1.format(*string)
        else:
            table += line2.format(*string)
        i += 1
    
    table += "    \\bottomrule\n"
    table += "\\end{tabular}"
    return table

def table_CS_summary(
    e="2xCO2",
    RHs=cst["RHs"],
    O3="a",
    O3c="f",
    ws=cst["ws"],
    prefix=cst["prefix"],
    base=cst["base"],
):
    RHsv = []
    for RH in RHs:
        if isinstance(RH, str):
            RHsv.append(cst["RHs_s"][RH])
        else:
            RHsv.append(RH)
    data = create_basic_data_repo(
        vs=["T_s", "φ_t"], es=["pi-control"] + [e], RHs=RHsv, O3s=[O3, O3c], ws=ws,
        prefix=prefix, base=base,
    )
    csdata = create_FFCS_data_repo(
        data, es=[e], RHs=RHsv, O3s=[O3, O3c], ws=ws,
    )
    table = table_CS_data(csdata, e=e, RHs=RHsv, O3=O3, O3c=O3c, ws=ws)
    return table

def table_CS_data(
    data,
    e="2xCO2",
    RHs=cst["RHs"],
    O3="a",
    O3c="f",
    ws=cst["ws"],
):
    RHss = [cst["RHs_v"][RH] for RH in RHs]
    line = "    {5}{0:4.2f} &{5}{1:5.2f} &{5}{2:5.2f} &{5}{3:5.2f} &{5}{4:5.2f}\\\\\n"
    table = (
        "\\begin{tabular}{@{}S[table-format=1.2]S[table-format=2.2]" +
        "S[table-format=2.2]cS[table-format=2.2]" +
        "S[table-format=2.2]@{}}\n"
    )
    table += "    \\toprule\n"
    table += "      &\multicolumn{4}{c}{$\Delta_{r}\ECS/\si{1}$}\\\\\n"
    table += "    \\cmidrule{2-5}\n"
    table += "    $w/\\si{{\mms}}$ &{{{0}}} &{{{1}}} &{{{2}}} &{{{3}}}".format(*RHss)
    table += "\\\\\n"
    table += "    \\midrule\n"
    for w in ws:
        if isinstance(w, float):
            string = (w,)
        else:
            string = (cst["ws_var_val"][w][O3], )
        string = string + tuple([
            (data["S"][e][RH][O3c][w] / data["S"][e][RH][O3][w]) - 1
            for RH in RHs
        ])
        if w == 0.2:
            string = string + ("\\color{red}", )
        else:
            string = string + ("", )
        table += line.format(*string)
    table += "    \\bottomrule\n"
    table += "\\end{tabular}"
    return table

def table_control_summary(
    RHs=cst["RHs"],
    prefix=cst["prefix"],
    base=cst["base"],
):
    RHsv = []
    for RH in RHs:
        if isinstance(RH, str):
            RHsv.append(cst["RHs_s"][RH])
        else:
            RHsv.append(RH)
    data = create_basic_data_repo(
        vs=["T_s", "φ_t"], es=["control", "pi-control"], RHs=RHsv,
        prefix=prefix, base=base,
    )
    table = table_control_data(data, RHs=RHsv)
    return table

def table_control_data(
    data,
    RHs=cst["RHs"]
):
    line = (
        "    {0:2s} &{1:5.2f} &{2:5.2f} &{3:4.2f} & &{4:6.2f} &{5:5.2f}" +
        "\\\\\n"
    )
    table = (
        "\\begin{tabular}{@{}cS[table-format=2.2]S[table-format=2.2]" +
        "S[table-format=1.2]cS[table-format=3.2]" +
        "S[table-format=1.2]@{}}\n"
    )
    table += "    \\toprule\n"
    table += (
        "      &$\\ntfp$ &$\\ntfc$ &$\\mathrm{Diff}$ & &$\\Tsfp$ " +
        "&$\\mathrm{Diff}$\\\\\n"
    )
    table += "    \\cmidrule{2-4} \\cmidrule{6-7}\n"
    table += (
        "    RH &\multicolumn{3}{c}{$/\si{\Wmc}$} & &\multicolumn{2}{c}{$/\si{\K}$}" +
        "\\\\\n"
    )
    table += "    \\midrule\n"
    for RH in RHs:
        string = (
            cst["RHs_v"][RH],
            data["φ_t"]["pi-control"][RH][-10:].mean(),
            data["φ_t"]["control"][RH][-10:].mean(),
            (
                data["φ_t"]["pi-control"][RH][-10:].mean() -
                data["φ_t"]["control"][RH][-10:].mean()
            ),
            data["T_s"]["pi-control"][RH][-10:].mean(),
            (
                data["T_s"]["pi-control"][RH][-10:].mean() -
                data["T_s"]["control"][RH][-10:].mean()
            ),
        )
        table += line.format(*string)
    table += "    \\bottomrule\n"
    table += "\\end{tabular}"
    return table

def table_summary(
    e="2xCO2",
    RH=0.4,
    O3="f",
    ws=cst["ws"],
    prefix=cst["prefix"],
    base=cst["base"],
):
    RHv = RH
    if isinstance(RH, str):
        RHv = cst["RHs_s"][RH]
    data = create_basic_data_repo(
        vs=["T_s", "φ_t"], es=["pi-control"] + [e], RHs=[RHv], O3s=[O3], ws=ws,
        prefix=prefix, base=base,
    )
    csdata = create_FFCS_data_repo(
        data, es=[e], RHs=[RHv], O3s=[O3], ws=ws,
    )
    table = table_data(csdata, e=e, RH=RHv, O3=O3, ws=ws)
    return table

def table_data(data, e="2xCO2", RH=0.4, O3="f", ws=cst["ws"]):
    line = (
        "    {7}{0:4.3f} &{7}{1:5.2f} &{7}{2:5.2f} &{7}{3:5.2f} & &{7}{4:6.2f} " +
        "&{7}{5:4.2f} &{7}{6:5.2f} \\\\\n"
    )
    table = (
        "\\begin{tabular}{@{}S[table-format=1.3]S[table-format=2.2]" +
        "S[table-format=1.2]S[table-format=1.2]cS[table-format=3.2]" +
        "S[table-format=1.2]S[table-format=1.2]@{}}\n"
    )
    table += "    \\toprule\n"
    table += "      &$\\ntf$ &$\\Ntf$ &$\\Fo$ & &$\\Tsf$ &$\\ECS$ & \\\\\n"
    table += "    \\cmidrule{2-3} \\cmidrule{5-6} \\cmidrule{8-9}\n"
    table += (
        "    $w/\\si{\\mms}$ &\\multicolumn{3}{c}{$/\\si{\Wmc}$} & " +
        "&\multicolumn{2}{c}{$/\\si{\\K}$} &$\\lambda/\\si{\\WmcK}$\\\\\n"
    )
    table += "    \\midrule\n"
    for w in ws:
        if isinstance(w, float):
            string = (w,)
        else:
            string = (cst["ws_var_val"][w][O3], )
        string = string + (
            data["φ_t,f"][e][RH][O3][w],
            data["N_f"][e][RH][O3][w],
            data["F_eff,f"][e][RH][O3][w],
            data["T_s,f"][e][RH][O3][w],
            data["S"][e][RH][O3][w],
            data["λ"][e][RH][O3][w],
        )
        if w == 0.2:
            string = string + ("\\color{red}", )
        else:
            string = string + ("", )
        table += line.format(*string)
    table += "    \\bottomrule\n"
    table += "\\end{tabular}"
    return table

""" Create data structure
"""

def create_data_repo(
    vs=cst["vars"],
    csvs=cst["csvars"],
    dvs=cst["dvars"],
    es=cst["exps"],
    cses=["2xCO2",],
    RHs=cst["RHs"],
    O3s=cst["O3s"],
    ws=cst["ws"],
    prefix=cst["prefix"],
    base=cst["base"],
):
    basic = create_basic_data_repo(
        vs=vs, es=es, RHs=RHs, O3s=O3s, ws=ws, prefix=prefix, base=base,
    )
    cs = create_FFCS_data_repo(
        basic, csvs=csvs, es=cses, RHs=RHs, O3s=O3s, ws=ws,
    )
    derived = create_derived_data_repo(
        basic, dvs=dvs, es=es, RHs=RHs, O3s=O3s, ws=ws,
    )
    data = {**basic, **cs, **derived}
    return data

def create_FFCS_data_repo(
    basic,
    es=["2xCO2"],
    RHs=cst["RHs"],
    O3s=cst["O3s"],
    ws=cst["ws"],
):
    data = create_empty_store(vs=cst["csvars"], es=es, RHs=RHs, O3s=O3s, ws=ws)
    for e in data[cst["csvars"][0]]:
        for RH in data[cst["csvars"][0]][e]:
            for O3 in data[cst["csvars"][0]][e][RH]:
                for w in data[cst["csvars"][0]][e][RH][O3]:
                    Ts = basic["T_s"][e][RH][O3][w]
                    Tsf = Ts[-10:].mean()
                    Ts = Ts - basic["T_s"]["pi-control"][RH][-10:].mean()
                    Nt = basic["φ_t"][e][RH][O3][w]
                    Nt = Nt - basic["φ_t"][e][RH][O3][w][-10:].mean()
                    phitf = basic["φ_t"][e][RH][O3][w][-10:].mean()
                    Ntf = phitf - basic["φ_t"]["pi-control"][RH][-10:].mean()
                    n0, n1 = clima.NT_slice_konrad(Ts, Nt, nxCO2=cst["exps"][e]["CO2"])
                    if O3 == "a":
                        n0 -= 0
                    (
                        data["S"][e][RH][O3][w],
                        data["F_eff,f"][e][RH][O3][w],
                        data["λ"][e][RH][O3][w],
                    ) = clima.ECS(
                        Ts[n0:n1], Nt[n0:n1], nxCO2=cst["exps"][e]["CO2"], konrad=True
                    )
                    data["ΔT_s"][e][RH][O3][w] = Ts
                    data["N"][e][RH][O3][w] = Nt
                    data["N_f"][e][RH][O3][w] = Ntf
                    data["F_eff"][e][RH][O3][w] = Ntf + data["F_eff,f"][e][RH][O3][w]
                    data["φ_t,f"][e][RH][O3][w] = phitf
                    data["T_s,f"][e][RH][O3][w] = Tsf
    return data

def create_derived_data_repo(
    basic,
    dvs=cst["dvars"],
    es=cst["exps"],
    RHs=cst["RHs"],
    O3s=cst["O3s"],
    ws=cst["ws"],
):
    dvsd = {key: cst["dvars"][key] for key in dvs}
    data = create_empty_store(vs=dvs, es=es, RHs=RHs, O3s=O3s, ws=ws)
    for v in data:
        for e in data[v]:
            for RH in data[v][e]:
                if data[v][e][RH] == None:
                    data[v][e][RH] = calculate_derived(basic, v, dvsd, e, RH)
                else:
                    for O3 in data[v][e][RH]:
                        for w in data[v][e][RH][O3]:
                            data[v][e][RH][O3][w] = calculate_derived(
                                basic, v, dvsd, e, RH, O3, w
                            )
    return data

def calculate_derived(
    store,
    v,
    vs,
    *args
):
    calc = []
    for param in vs[v]["args"]:
        if param in ["p", "ph"]:
            calc.append(store[param])
        else:
            temp = store[param]
            for arg in args:
                temp = temp[arg]
            calc.append(temp)
    calc = tuple(calc)
    calc = vs[v]["function"](*calc)
    if v in ["Γ", "Γ_m"]:
        calc = -calc
    return calc

def create_basic_data_repo(
    vs=cst["vars"],
    es=cst["exps"],
    RHs=cst["RHs"],
    O3s=cst["O3s"],
    ws=cst["ws"],
    prefix=cst["prefix"],
    base=cst["base"],
):
    data = create_empty_store(vs=vs, es=es, RHs=RHs, O3s=O3s, ws=ws)
    for v in data:
        if data[v] == None:
            data[v] = load_variable_from_file(
                v, prefix=prefix, base=base,
            )
        else:
            for e in data[v]:
                for RH in data[v][e]:
                    if data[v][e][RH] == None:
                        data[v][e][RH] = load_variable_from_file(
                            v, prefix=prefix, base=base, RH=RH, exp=e,
                        )
                    else:
                        for O3 in data[v][e][RH]:
                            for w in data[v][e][RH][O3]:
                                data[v][e][RH][O3][w] = load_variable_from_file(
                                    v, prefix=prefix, base=base,
                                    RH=RH, O3=O3, w=w, exp=e,
                                )
    return data

def create_empty_store(
    vs=cst["vars"],
    es=cst["exps"],
    RHs=cst["RHs"],
    O3s=cst["O3s"],
    ws=cst["ws"],
):
    store = {v: {} for v in vs}
    for v in store:
        if v in ["p", "ph"]:
            store[v] = None
        else:
            for e in es:
                if e in cst["cntrls"]:
                    store[v][e] = {RH: None for RH in RHs}
                else:
                    store[v][e] = {
                        RH: {O3: {w: None for w in ws} for O3 in O3s} for RH in RHs
                    }
    return store

def load_variable_from_file(
    var,
    prefix=cst["prefix"],
    base=cst["base"],
    RH=0.4,
    O3="a",
    w=0.2,
    cnv="h",
    exp="pi-control",
):
    path = path_konrad(
        prefix=prefix, base=base, RH=RH, O3=O3, w=w, cnv=cnv, exp=exp,
    )
    if var in cst["vars"]:
        with clima.ncload(path, mode="r") as data:
            if cst["vars"][var]["group"] == "":
                temp = data.variables[cst["vars"][var]["name"]][...].data
            else:
                if var in ["H"]:
                    if exp in ["control"]:
                        temp = 0
                    else:
                        temp = data.groups[cst["vars"][var]["group"]]
                        temp = temp.variables[cst["vars"][var]["name"]][...]
                        temp = temp.data.item()
                else:
                    temp = data.groups[cst["vars"][var]["group"]]
                    temp = temp.variables[cst["vars"][var]["name"]][...]
                    temp = temp.data
                    if var in ["φ_sw↑", "φ_lw↑", "Q_up"]:
                        temp = -temp
        return temp

def path_konrad(
    prefix=cst["prefix"],
    base=cst["base"],
    RH=0.4,
    O3="a",
    w=0.2,
    cnv="h",
    exp="pi-control",
):
    path = f"{base}/{prefix}"
    path += f"{RH * 100:03.0f}-"
    path += f"{cst['exps'][exp]['CO2'] * 10:03.0f}_"
    if isinstance(w, float):
        path += f"{O3:s}{w * 1000:03.0f}{cnv:s}{cst['exps'][exp]['suffix']:s}.nc"
    else:
        path += f"{O3:s}{w:s}{cnv:s}{cst['exps'][exp]['suffix']:s}.nc"
    return path