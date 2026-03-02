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


ciclos_05 = glob("NFx1_CPAx2/**/**/*ciclo_promedio_H_M.txt")
resultados_05 = glob("NFx1_CPAx2/**/**/*resultados.txt")
ciclos_10 = glob("NFx2_CPAx1/**/**/*ciclo_promedio_H_M.txt")
resultados_10 = glob("NFx2_CPAx1/**/**/*resultados.txt")
for p in ciclo_05:
    print('  ',p)
for p in ciclos_10:
    print('  ',p)

print('\n')
for res in resultados_05:
    print('  ',res)
for res in resultados_10:
    print('  ',res)


#%%

fig, (ax,ax2) =plt.subplots(1,2,figsize=(10,5),constrained_layout=True,sharey=True)

for i,e in enumerate(ciclos_05):
    _,_,_, H_05,M_05,_ = lector_ciclos(ciclos_05[i])
    ax.plot(H_05/1000,M_05,'-',label='05')

for i,e in enumerate(ciclos_10):
    _,_,_, H_10,M_10,_ = lector_ciclos(ciclos_10[i])
    ax2.plot(H_10/1000,M_10,'-',label='10')
# ax.plot(H_top/1000,M_top,'-',label='top')
# ax.plot(H_center/1000,M_center,'-',label='center')
# ax.plot(H_bottom/1000,M_bottom,'-',label='bottom')



ax.set_ylabel('M (A/m)')
for a in ax,ax2:
    a.grid()
    a.set_xlabel('H (kA/m)')
    a.legend()
#plt.savefig('MvsH_NF@cit_300kHz_57kAm_top_center_bottom.png',dpi=300)

#%% Resultados


res_top = ResultadosESAR(os.path.dirname(dir_res_top[0]))

res_center = ResultadosESAR(os.path.dirname(dir_res_center[0]))

res_bottom = ResultadosESAR(os.path.dirname(dir_res_bottom[0]))

#%% Temp vs time
dt_top = res_top.time[-1]-res_top.time[0]
dt_center = res_center.time[-1]-res_center.time[0]
dt_bottom = res_bottom.time[-1]-res_bottom.time[0]
dT_top = res_top.temperatura[-1]-res_top.temperatura[0]
dT_center = res_center.temperatura[-1]-res_center.temperatura[0]
dT_bottom = res_bottom.temperatura[-1]-res_bottom.temperatura[0]

fig0, ax = plt.subplots(figsize=(8,5),constrained_layout=True)
ax.plot(res_top.time,res_top.temperatura,'.-',label=rf'''top
$\Delta$t={dt_top:.2f} s 
$\Delta$T={dT_top:.2f} °C
rate = {dT_top/dt_top:.2f} °C/s
''')
ax.plot(res_center.time,res_center.temperatura,'.-',label=f'''center 
$\Delta$t={dt_center:.2f} s 
$\Delta$T={dT_center:.2f} °C
rate = {dT_center/dt_center:.2f} °C/s
''')
ax.plot(res_bottom.time,res_bottom.temperatura,'.-',label=f'''bottom
$\Delta$t={dt_bottom:.2f} s 
$\Delta$T={dT_bottom:.2f} °C
rate = {dT_bottom/dt_bottom:.2f} °C/s
''')
ax.grid()
ax.set_xlabel('t (s)')
ax.set_ylabel('T (°C)')
ax.legend()
plt.suptitle('Templogs\nNF@cit - 300 kHz & 57 kA/m',fontsize=14)
plt.savefig('templogs_NF@cit_300kHz_57kAm_top_center_bottom.png',dpi=300)
#%% tau y SAR vs time/Temp
fig1, ((ax,ax3),(ax2,ax4)) = plt.subplots(2,2,figsize=(11,7),constrained_layout=True,sharex='col',sharey='row')

ax.set_title('τ vs t',loc='left')
ax.plot(res_top.time,res_top.tau,'.-',label='top')
ax.plot(res_center.time,res_center.tau,'.-',label='center')
ax.plot(res_bottom.time,res_bottom.tau,'.-',label='bottom')
ax.set_ylabel('τ (ns)')

ax2.set_title('SAR vs t',loc='left')
ax2.plot(res_top.time,res_top.SAR,'.-',label='top')
ax2.plot(res_center.time,res_center.SAR,'.-',label='center')
ax2.plot(res_bottom.time,res_bottom.SAR,'.-',label='bottom')
ax2.set_ylabel('SAR (W/g)')
ax2.set_xlabel('t (s)')

ax3.set_title('τ vs T',loc='left')
ax3.plot(res_top.temperatura,res_top.tau,'.-',label='top')
ax3.plot(res_center.temperatura,res_center.tau,'.-',label='center')
ax3.plot(res_bottom.temperatura,res_bottom.tau,'.-',label='bottom')

ax4.set_title('SAR vs T',loc='left')
ax4.plot(res_top.temperatura,res_top.SAR,'.-',label='top')
ax4.plot(res_center.temperatura,res_center.SAR,'.-',label='center')
ax4.plot(res_bottom.temperatura,res_bottom.SAR,'.-',label='bottom')
ax4.set_xlabel('T (s)')

plt.suptitle('tau & SAR vs time/Temp\nNF@cit - 300 kHz & 57 kA/m',fontsize=15)

for a in ax,ax2,ax3,ax4:
    a.grid()
    a.legend()
plt.savefig('tau_SAR_vs_time_temp_NF@cit_300kHz_57kAm_top_center_bottom.png',dpi=300)
plt.show()


#%% dphi y mag vs time / Temp
fig2, ((ax,ax3),(ax2,ax4)) = plt.subplots(2,2,figsize=(11,7),constrained_layout=True,sharex='col',sharey='row')

ax.set_title('dphi vs t',loc='left')
ax.plot(res_top.time,res_top.dphi_fem,'.-',label='top')
ax.plot(res_center.time,res_center.dphi_fem,'.-',label='center')
ax.plot(res_bottom.time,res_bottom.dphi_fem,'.-',label='bottom')
ax.set_ylabel('dphi (rad)')

ax2.set_title('Magnitud vs t',loc='left')
ax2.plot(res_top.time,res_top.magnitud_fund,'.-',label='top')
ax2.plot(res_center.time,res_center.magnitud_fund,'.-',label='center')
ax2.plot(res_bottom.time,res_bottom.magnitud_fund,'.-',label='bottom')
ax2.set_ylabel('magnitud ()')
ax2.set_xlabel('t (s)')

ax3.set_title('dphi vs Temp',loc='left')
ax3.plot(res_top.temperatura,res_top.dphi_fem,'.-',label='top')
ax3.plot(res_center.temperatura,res_center.dphi_fem,'.-',label='center')
ax3.plot(res_bottom.temperatura,res_bottom.dphi_fem,'.-',label='bottom')

ax4.set_title('Magnitud vs Temp',loc='left')
ax4.plot(res_top.temperatura,res_top.magnitud_fund,'.-',label='top')
ax4.plot(res_center.temperatura,res_center.magnitud_fund,'.-',label='center')
ax4.plot(res_bottom.temperatura,res_bottom.magnitud_fund,'.-',label='bottom')
ax4.set_xlabel('T (°C)')

plt.suptitle('dphi & manitud vs time/Temp\nNF@cit - 300 kHz & 57 kA/m',fontsize=14)

for a in ax,ax2,ax3,ax4:
    a.grid()
    a.legend()
plt.savefig('dphi_mag_vs_time_temp_NF@cit_300kHz_57kAm_top_center_bottom.png',dpi=300)
plt.show()

#%% Magnetizacion maxima vs time / Temp

fig3,(ax,ax2) =plt.subplots(1,2,figsize=(12,5),constrained_layout=True,sharey='row')

ax.plot(res_top.time,res_top.mag_max,'.-',label='top')
ax.plot(res_center.time,res_center.mag_max,'.-',label='center')
ax.plot(res_bottom.time,res_bottom.mag_max,'.-',label='bottom')

ax2.plot(res_top.temperatura,res_top.mag_max,'.-',label='top')
ax2.plot(res_center.temperatura,res_center.mag_max,'.-',label='center')
ax2.plot(res_bottom.temperatura,res_bottom.mag_max,'.-',label='bottom')

ax.grid()
ax2.grid()
ax.set_xlabel('t (s)')
ax2.set_xlabel('T (°C)')
ax.set_ylabel('Mmax (A/m)')

ax.legend()
ax2.legend()

plt.suptitle('Mmax vs time/Temp\nNF@cit - 300 kHz & 57 kA/m',fontsize=14)
plt.savefig('Mmax_vs_time_temp_NF@cit_300kHz_57kAm_top_center_bottom.png',dpi=300)
plt.show()




# %%
