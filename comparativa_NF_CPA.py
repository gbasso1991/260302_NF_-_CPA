#%% Librerias y paquetes 
import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import chardet
import re
from clase_resultados import ResultadosESAR
#%% LEctor de resultados
def lector_resultados(path):
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']

    # Leer las primeras 20 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                # Patrón para valores con incertidumbre (ej: 331.45+/-6.20 o (9.74+/-0.23)e+01)
                match_uncertain = re.search(r'(.+)_=_\(?([-+]?\d+\.\d+)\+/-([-+]?\d+\.\d+)\)?(?:e([+-]\d+))?', line)
                if match_uncertain:
                    key = match_uncertain.group(1)[2:]  # Eliminar '# ' al inicio
                    value = float(match_uncertain.group(2))
                    uncertainty = float(match_uncertain.group(3))
                    
                    # Manejar notación científica si está presente
                    if match_uncertain.group(4):
                        exponent = float(match_uncertain.group(4))
                        factor = 10**exponent
                        value *= factor
                        uncertainty *= factor
                    
                    meta[key] = ufloat(value, uncertainty)
                else:
                    # Patrón para valores simples (sin incertidumbre)
                    match_simple = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                    if match_simple:
                        key = match_simple.group(1)[2:]
                        value = float(match_simple.group(2))
                        meta[key] = value
                    else:
                        # Capturar los casos con nombres de archivo
                        match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                        if match_files:
                            key = match_files.group(1)[2:]
                            value = match_files.group(2)
                            meta[key] = value

    # Leer los datos del archivo (esta parte permanece igual)
    data = pd.read_table(path, header=15,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)

    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)

    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)

    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N
#%% LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "pendiente_HvsI ": float(lines[3].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}

    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype={'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m

    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata
#%% Obtengo ciclos y resultados para cada concentracion - Todo a 300 kHz
# 5 g/L
ciclos_05 = glob("NFx1_CPAx2/**/**/*ciclo_promedio_H_M.txt") 
resultados_05 = glob("NFx1_CPAx2/**/**/*resultados.txt")
ciclos_05.sort()
resultados_05.sort()

# 10 g/L
ciclos_10 = glob("NFx2_CPAx1/**/**/*ciclo_promedio_H_M.txt")
resultados_10 = glob("NFx2_CPAx1/**/**/*resultados.txt")
ciclos_10.sort()
resultados_10.sort()

for p in ciclos_05:
    print('  ',p)
for p in ciclos_10:
    print('  ',p)

print('\n')
for res in resultados_05:
    print('  ',res)
for res in resultados_10:
    print('  ',res)

#%% Ploteo Ciclos Promedio 

fig0, (ax,ax2) =plt.subplots(1,2,figsize=(10,5),constrained_layout=True,sharey=True)

for i,e in enumerate(ciclos_05):
    _,_,_, H_05,M_05,_ = lector_ciclos(ciclos_05[i])
    ax.plot(H_05/1000,M_05,'-',label=f'NF{i}')

for i,e in enumerate(ciclos_10):
    _,_,_, H_10,M_10,_ = lector_ciclos(ciclos_10[i])
    ax2.plot(H_10/1000,M_10,'-',label=f'NF{i+3}')

ax.set_ylabel('M (A/m)')
ax.set_title('NF - 5 g/L',loc='center')
ax2.set_title('NF - 10 g/L',loc='center')
for a in ax,ax2:
    a.grid()
    a.set_xlabel('H (kA/m)')
    a.legend()
plt.suptitle('Comparativa ciclos promedio\n300 kHz - 57 kA/m')
#%% Listas con Resultados
res_05=[]
for r in resultados_05:
    res_05.append(ResultadosESAR(os.path.dirname(r)))
res_10=[]
for r in resultados_10:
    res_10.append(ResultadosESAR(os.path.dirname(r)))

#%% 1- Templogs y ecSAR
conc_05 = 5 #g/L
conc_10 = 10 #g/L
ecSAR_05=[]
ecSAR_10=[]
rates_05=[]
rates_10=[]
fig1, (ax,ax2) =plt.subplots(2,1,figsize=(10,6),constrained_layout=True,sharey=True,sharex=True)

for i,r in enumerate(res_05):
    dt = r.time[-1]-r.time[0]
    dT = r.temperatura[-1]-r.temperatura[0]
    rate=dT/dt
    rates_05.append(rate)
    ecSAR_05.append(rate*4186/conc_05)
    label=f'$\Delta$t={dt:.2f} s  $\Delta$T={dT:.2f} °C  WR= {rate:.2f} °C/s'
    ax.plot(r.time,r.temperatura,'.-',label=label)

for i,r in enumerate(res_10):
    dt = r.time[-1]-r.time[0]
    dT = r.temperatura[-1]-r.temperatura[0]
    rate=dT/dt
    rates_10.append(rate)
    ecSAR_10.append(rate*4186/conc_10)
    label=f'$\Delta$t={dt:.2f} s  $\Delta$T={dT:.2f} °C  WR= {rate:.2f} °C/s'
    ax2.plot(r.time,r.temperatura,'.-',label=label)

for a in ax,ax2:
    a.grid()
    a.legend(loc='upper left')
    a.set_ylabel('T (ºC)')
rate_05=ufloat(np.mean(rates_05),np.std(rates_05))
rate_10=ufloat(np.mean(rates_10),np.std(rates_10))

ecSAR_05=ufloat(np.mean(ecSAR_05),np.std(ecSAR_05))
ecSAR_10=ufloat(np.mean(ecSAR_10),np.std(ecSAR_10))

ax.text(0.98,0.1,f'WR = {rate_05:.1uS} °C/s\necSAR = {ecSAR_05:.2uS} W/g',
        bbox=dict(boxstyle="round", fc='C3',alpha=0.6,lw=1),
        ha='right',va='bottom',
        transform=ax.transAxes)

ax2.text(0.98,0.1,f'WR = {rate_10:.1uS} °C/s\necSAR = {ecSAR_10:.2uS} W/g',
        bbox=dict(boxstyle="round", fc='C3',alpha=0.6,lw=1),
        ha='right',va='bottom',
        transform=ax2.transAxes)

ax.set_xlim(0,)
ax.set_title('NF - 5 g/L',loc='left')
ax2.set_title('NF - 10 g/L',loc='left')    
ax2.set_xlabel('t (s)')
plt.suptitle('Templogs NF - 5 g/L & 10 g/L')

#%% 2 - Tau vs time / Temp
fig20, (ax,ax2) =plt.subplots(2,1,figsize=(10,6),constrained_layout=True,sharey=True,sharex=True)

for i,r in enumerate(res_05):
    ax.plot(r.time,r.tau,'.-',label=f'NF {i}')

for i,r in enumerate(res_10):
    ax2.plot(r.time,r.tau,'.-',label=f'NF {i}')

for a in ax,ax2:
    a.grid()
    a.legend(loc='lower left')
    a.set_ylabel('τ (ns)')
ax.set_xlim(0,)
ax.set_title('NF - 5 g/L',loc='left')
ax2.set_title('NF - 10 g/L',loc='left')    
ax2.set_xlabel('t (s)')
plt.suptitle('tau NF vs time - 5 g/L & 10 g/L')

fig21, (ax,ax2) =plt.subplots(2,1,figsize=(10,6),constrained_layout=True,sharey=True,sharex=True)

for i,r in enumerate(res_05):
    ax.plot(r.temperatura,r.tau,'.-',label=f'NF {i}')

for i,r in enumerate(res_10):
    ax2.plot(r.temperatura,r.tau,'.-',label=f'NF {i}')

for a in ax,ax2:
    a.grid()
    a.legend(loc='upper right')
    a.set_ylabel('τ (ns)')
ax.set_xlim(24,75)
ax.set_title('NF - 5 g/L',loc='left')
ax2.set_title('NF - 10 g/L',loc='left')    
ax2.set_xlabel('T (°C)')
plt.suptitle('tau NF vs temperatura - 5 g/L & 10 g/L')

#%% 3 - SAR vs time / Temp

ESAR_05_all=ufloat(np.mean(np.concatenate([r.SAR for r in res_05])),np.std(np.concatenate([r.SAR for r in res_05])))
ESAR_10_all=ufloat(np.mean(np.concatenate([r.SAR for r in res_10])),np.std(np.concatenate([r.SAR for r in res_10])))

text_05 = f'ESAR = {ESAR_05_all:.2uS} W/g'
text_10 = f'ESAR = {ESAR_10_all:.2uS} W/g'
fig30, (ax,ax2) =plt.subplots(2,1,figsize=(10,6),constrained_layout=True,sharey=True,sharex=True)

for i,r in enumerate(res_05):
    ax.plot(r.time,r.SAR,'.-',label=f'NF {i}')

for i,r in enumerate(res_10):
    ax2.plot(r.time,r.SAR,'.-',label=f'NF {i}')


ax.text(0.95,0.9,f'ESAR = {ESAR_05_all:.2uS} W/g',
        bbox=dict(boxstyle="round", fc='C3',alpha=0.6,lw=1),
        ha='right',va='top',
        transform=ax.transAxes)

ax2.text(0.95,0.9,f'ESAR = {ESAR_10_all:.2uS} W/g',
        bbox=dict(boxstyle="round", fc='C3',alpha=0.6,lw=1),
        ha='right',va='top',
        transform=ax2.transAxes)


for a in ax,ax2:
    a.grid()
    #a.legend(loc='upper right')
    a.set_ylabel('SAR (W/g)')
ax.set_xlim(0,)
ax.set_title('NF - 5 g/L',loc='left')
ax2.set_title('NF - 10 g/L',loc='left')    
ax2.set_xlabel('t (s)')
plt.suptitle('SAR NF vs time - 5 g/L & 10 g/L')

fig31, (ax,ax2) =plt.subplots(2,1,figsize=(10,6),constrained_layout=True,sharey=True,sharex=True)

for i,r in enumerate(res_05):
    ax.plot(r.temperatura,r.SAR,'.-',label=f'NF {i}')

for i,r in enumerate(res_10):
    ax2.plot(r.temperatura,r.SAR,'.-',label=f'NF {i}')


ax.text(0.95,0.9,f'ESAR = {ESAR_05_all:.2uS} W/g',
        bbox=dict(boxstyle="round", fc='C3',alpha=0.6,lw=1),
        ha='right',va='top',
        transform=ax.transAxes)

ax2.text(0.95,0.9,f'ESAR = {ESAR_10_all:.2uS} W/g',
        bbox=dict(boxstyle="round", fc='C3',alpha=0.6,lw=1),
        ha='right',va='top',
        transform=ax2.transAxes)


for a in ax,ax2:
    a.grid()
    #a.legend(loc='upper right')
    a.set_ylabel('SAR (W/g)')
ax.set_xlim(24,75)
ax.set_title('NF - 5 g/L',loc='left')
ax2.set_title('NF - 10 g/L',loc='left')    
ax2.set_xlabel('T (°C)')
plt.suptitle('SAR NF vs temperatura - 5 g/L & 10 g/L')


#%% 4 - Hc vs time/Temp

fig40, (ax,ax2) =plt.subplots(2,1,figsize=(10,6),constrained_layout=True,sharey=True,sharex=True)

for i,r in enumerate(res_05):
    ax.plot(r.time,r.Hc,'.-',label=f'NF {i}')

for i,r in enumerate(res_10):
    ax2.plot(r.time,r.Hc,'.-',label=f'NF {i}')

for a in ax,ax2:
    a.grid()
    a.legend(loc='upper right')
    a.set_ylabel('Hc (A/m)')
ax.set_xlim(0,)
ax.set_title('NF - 5 g/L',loc='left')
ax2.set_title('NF - 10 g/L',loc='left')    
ax2.set_xlabel('t (s)')
plt.suptitle('Hc NF vs time - 5 g/L & 10 g/L')

fig41, (ax,ax2) =plt.subplots(2,1,figsize=(10,6),constrained_layout=True,sharey=True,sharex=True)

for i,r in enumerate(res_05):
    ax.plot(r.temperatura,r.Hc,'.-',label=f'NF {i}')

for i,r in enumerate(res_10):
    ax2.plot(r.temperatura,r.Hc,'.-',label=f'NF {i}')

for a in ax,ax2:
    a.grid()
    a.legend(loc='upper right')
    a.set_ylabel('Hc (A/m)')
ax.set_xlim(24,75)
ax.set_title('NF - 5 g/L',loc='left')
ax2.set_title('NF - 10 g/L',loc='left')    
ax2.set_xlabel('T (°C)')
plt.suptitle('Hc NF vs temperatura - 5 g/L & 10 g/L')
#%% 5 - Mag Max vs time/Temp

fig50, (ax,ax2) =plt.subplots(2,1,figsize=(10,6),constrained_layout=True,sharey=False,sharex=True)

for i,r in enumerate(res_05):
    dM = r.mag_max[-1]-r.mag_max[0]
    label=f'NF{i} - $\Delta$'+'M$_{max}$ = '+f'{dM:.1f} A/m '
    ax.plot(r.time,r.mag_max,'.-',label=label)

for i,r in enumerate(res_10):
    dM = r.mag_max[-1]-r.mag_max[0]
    label=f'NF{i} - $\Delta$'+'M$_{max}$ = '+f'{dM:.1f} A/m '
    ax2.plot(r.time,r.mag_max,'.-',label=label)

for a in ax,ax2:
    a.grid()
    a.legend(loc='best')
    a.set_ylabel('M$_{max}$ (A/m)')
ax.set_xlim(0,)
ax.set_title('NF - 5 g/L',loc='left')
ax2.set_title('NF - 10 g/L',loc='left')    
ax2.set_xlabel('t (s)')
plt.suptitle('M$_{max}$ NF vs time - 5 g/L & 10 g/L')


fig51, (ax,ax2) =plt.subplots(2,1,figsize=(10,6),constrained_layout=True,sharey=False,sharex=True)

for i,r in enumerate(res_05):
    dM = r.mag_max[-1]-r.mag_max[0]
    label=f'NF{i} - $\Delta$'+'M$_{max}$ = '+f'{dM:.1f} A/m '
    ax.plot(r.temperatura,r.mag_max,'.-',label=label)

for i,r in enumerate(res_10):
    dM = r.mag_max[-1]-r.mag_max[0]
    label=f'NF{i} - $\Delta$'+'M$_{max}$ = '+f'{dM:.1f} A/m '
    ax2.plot(r.temperatura,r.mag_max,'.-',label=label)

for a in ax,ax2:
    a.grid()
    a.legend(loc='upper right')
    a.set_ylabel('M$_{max}$ (A/m)')
ax.set_xlim(24,75)
ax.set_title('NF - 5 g/L',loc='left')
ax2.set_title('NF - 10 g/L',loc='left')    
ax2.set_xlabel('T (°C)')
plt.suptitle('M$_{max}$ NF vs temperatura - 5 g/L & 10 g/L')


# %% Salvo las figuras 
figs=[fig0,fig1,fig20,fig21,fig30,fig31,fig40,fig41,fig50,fig51]
names=['ciclos_promedio_NF5_NF10',
       'templog_NF5_NF10',
       'tau_vs_time_NF5_NF10',
       'tau_vs_Temp_NF5_NF10',
       'ESAR_vs_time_NF5_NF10',
       'ESAR_vs_Temp_NF5_NF10',
       'Hc_vs_time_NF5_NF10',
       'Hc_vs_Temp_NF5_NF10',
       'Mmax_vs_time_NF5_NF10',
       'Mmax_vs_Temp_NF5_NF10' ]


for i,e in enumerate(zip(figs,names)):
    e[0].savefig(f'{i}_{e[1]}.png',dpi=300)

# %%
