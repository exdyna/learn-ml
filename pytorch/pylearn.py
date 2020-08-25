from __future__ import print_function
import torch

x=torch.empty(5,3)



# import numpy as np
# import os
# import itertools
# import pandas as pd
# import shutil
# import math
# import csv
# from scipy import optimize
# import time
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

# def setup(ax):
#     ax.grid(True)
#     for tick in ax.xaxis.get_major_ticks():
#         tick.label.set_fontsize(font_size)
#     for tick in ax.yaxis.get_major_ticks():
#         tick.label.set_fontsize(font_size)
        
#     # ax.xaxis.set_major_locator(ticker.MultipleLocator(100.0))
#     # ax.xaxis.set_minor_locator(ticker.MultipleLocator(50.0))

#     ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
#     ax.yaxis.set_major_locator(ticker.MaxNLocator(6))

#     ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,4), useOffset=True, useLocale=None, useMathText=True)
#     ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,4), useOffset=True, useLocale=None, useMathText=True)


# # define the objective function

# def fun_obj(x):

#     # the optimiation does not deal with damage parameters
#     job_id = int(time.time())

#     fid_fn_job_para_inp = open(fn_job_para_inp,'w')

#     fid_fn_job_para_inp.write("*parameter\n")
#     fid_fn_job_para_inp.write("Density = %.8e\n" % Density)
#     fid_fn_job_para_inp.write("Youngs_Modulus = %.8e\n" % x[0])
#     fid_fn_job_para_inp.write("Poisson_Ratio = %.8e\n" % x[1])
#     fid_fn_job_para_inp.write("JC_A = %.8e\n" % x[2])
#     fid_fn_job_para_inp.write("JC_B = %.8e\n" % x[3])
#     fid_fn_job_para_inp.write("JC_n = %.8e\n" % x[4])
#     fid_fn_job_para_inp.write("JC_m = %.8e\n" % x[5])
#     fid_fn_job_para_inp.write("JC_Tm = %.8e\n" % x[6])
#     fid_fn_job_para_inp.write("JC_Ts = %.8e\n" % x[7])
#     fid_fn_job_para_inp.write("damage_initiation_para = %.8e\n" % x[8])
#     fid_fn_job_para_inp.write("damage_evolution_para_1 = %.8e\n" % x[9])
#     fid_fn_job_para_inp.write("damage_evolution_para_2 = %.8e\n" % x[10])
#     fid_fn_job_para_inp.write("damage_max_degradation = %.8e\n" % x[11])
#     fid_fn_job_para_inp.write("damage_viscosity = %.8e\n" % 2.0)
#     fid_fn_job_para_inp.write("DT_mass_scaling = %.8e\n" % 1.0e-1)
#     fid_fn_job_para_inp.write("tip_depth = %.8e\n" % (tip_depth_nm*1e-3))
#     fid_fn_job_para_inp.write("tip_depth_nm = %.8e" % tip_depth_nm)
#     fid_fn_job_para_inp.close()

#     # ==================================================================================================
#     cmd_line = 'cp '+job_name+' job.inp'
#     os.system(cmd_line)

#     # run abaqus

#     cmd_line = 'abq2018 j=job cpus=16 ask_delete=off interactive'
#     os.system(cmd_line)
#     time.sleep(1)

#     # run abaqus cae to extract the history data
#     fn_tmp_data_file='data_rf_history.csv'
#     if os.path.isfile(fn_tmp_data_file):
#         os.remove(fn_tmp_data_file)
    
#     cmd_line = 'abq2018 cae nogui=py-read-history'
#     os.system(cmd_line)
#     time.sleep(1)

#     fn_sim_data_raw = data_folder+'/'+str(job_id)+'.csv'
#     cmd_line = 'mv '+fn_tmp_data_file+' '+ fn_sim_data_raw
#     os.system(cmd_line)
    

#     # ==================================================================================================
#     # # processing experimental and simulation data
#     # fn_exp_data_raw = current_folder+'/exp-al7050-new-tip.csv'
#     # exp_data_raw = np.loadtxt(fn_exp_data_raw, delimiter=',',dtype=float)

#     # condition = exp_data_raw[:,0] <= tip_depth_nm+10
#     # exp_data_raw = exp_data_raw[condition,:]

#     # exp_data_raw[:,1]=exp_data_raw[:,1]*1000
    
#     # exp_fit_coef = np.polyfit(exp_data_raw[:,0],exp_data_raw[:,1],2)
        
#     # sim_data = np.loadtxt(fn_sim_data_raw, delimiter=',',dtype=float)

#     # sim_fit_coef = np.polyfit(sim_data[:,0],sim_data[:,1],2)
    
#     # exp_data = np.zeros_like(sim_data)    
#     # exp_data[:,0] = sim_data[:,0]
#     # exp_data[:,1] = np.polyval(exp_fit_coef,exp_data[:,0])
#     # exp_data[0,0] = 0.0
#     # exp_data[0,1] = 0.0

#     # exp_data_grad = 2.0*exp_fit_coef[0]*exp_data[:,0] + exp_fit_coef[1]
#     # sim_data_grad = 2.0*sim_fit_coef[0]*sim_data[:,0] + sim_fit_coef[1]

#     # ==================================================================================================
#     # calculat the L2 error norm 
#     # obj_val = np.sqrt(np.sum(np.square((exp_data[:,1]-sim_data[:,1]))))

#     # obj_val = np.sqrt(np.sum(np.square((exp_data_grad[:,1]-sim_data_grad[:,1]))))
#     obj_val = 0.0
#     # ==================================================================================================
#     # record everything to workbook
#     fid_fn_job_workbook = open(fn_job_workbook,'a')

#     fid_fn_job_workbook.write("%d," % job_id)
#     fid_fn_job_workbook.write("%.8e," % obj_val)
#     # fid_fn_job_workbook.write("%.8e," % JC_A)

#     for xi in x:
#         fid_fn_job_workbook.write("%.8e," % xi)

#     fid_fn_job_workbook.write("\n")
#     fid_fn_job_workbook.close()
    

#     # ==================================================================================================
#     # # plot the data
#     # fn_figure = figure_folder+'/'+str(job_id)+'.jpg'

#     # plt.clf()
#     # fig = plt.figure(figsize=(9,3))
#     # ax= plt.subplot2grid((1,1),(0,0))
#     # ax.plot(exp_data_raw[:,0],exp_data_raw[:,1], color='black',linewidth=0,marker='o',markersize=marker_size/2,label='experiment')
#     # ax.plot(exp_data[:,0],exp_data[:,1], color='blue',linewidth=1,marker='o',markersize=marker_size/2,label='experiment fitting')
#     # ax.plot(sim_data[:,0],sim_data[:,1], color='red',linewidth=1,marker='o',markersize=marker_size,label='simulation')    

#     # ax.set_ylabel('Force ($\mu$N)',fontsize=font_size)
#     # ax.legend()
#     # setup(ax)

#     # ax.set_xlabel('Depth (nm)',fontsize=font_size)
#     # fig.align_ylabels()
#     # fig.subplots_adjust(hspace=0.5,top=0.9,bottom=0.2,left=0.2)
#     # plt.savefig(fn_figure,dpi = 200)
#     # plt.close()

#     cmd_line = 'clearabaqus.sh'
#     os.system(cmd_line)

#     time.sleep(1)

#     return obj_val


# # ==============================================================================
# # run optimization
# # ==============================================================================

# font_size = 10
# marker_size = 2

# current_folder = os.getcwd()

# job_folder = current_folder
# if not(os.path.exists(job_folder)):        
#     os.makedirs(job_folder)
# print(job_folder)

# data_folder = current_folder+'/run/data'
# if not(os.path.exists(data_folder)):        
#     os.makedirs(data_folder)
# print(data_folder)

# figure_folder = current_folder+'/run/figure'
# if not(os.path.exists(figure_folder)):        
#     os.makedirs(figure_folder)
# print(figure_folder)


# job_para_title = ('job_id,obj_val,E,v,A,B,n,m,Tm,Ts,dmg_ini,dmg_evol_1,dmg_evol_2,dmg_max_deg,rho,\n')

# #  write parameter list
# fn_job_workbook = current_folder+'/run/data/data-workbook.csv'
# if os.path.isfile(fn_job_workbook):
#     os.remove(fn_job_workbook)

# fid_fn_job_workbook = open(fn_job_workbook,'w')
# fid_fn_job_workbook.write(job_para_title)
# fid_fn_job_workbook.close()

# fn_job_para_inp = job_folder+'/include_parameter.inc'

# # ==============================================================================
# # define the initial value of the design variables
# # ==============================================================================
# # job_name = 'job-implicit-nodamage.inp'

# job_name = 'job-explicit-nodamage.inp'

# # tip depth (nm)
# tip_depth_nm= 200.0



# for i in range(1000):

#     Youngs_Modulus= np.random.random_sample()*(1500-1)+1 # E
#     Poisson_Ratio = np.random.random_sample()*(0.45-0.1)+0.1

#     JC_A = np.random.random_sample()*(5.0-0.001)+0.01
#     JC_B = np.random.random_sample()*(5.0-0.0)
#     JC_n = np.random.random_sample()*(5.0-0.0)

#     JC_m = 0.0
#     JC_Tm = 300.0
#     JC_Ts = 298.0    

#     damage_initiation_para = 0.0
#     damage_evolution_para_1 = 0.0
#     damage_evolution_para_2 = 0.0
#     damage_max_degradation=0.0

#     Density = 5.E-6

#     x0 = [Youngs_Modulus, Poisson_Ratio,JC_A,JC_B,JC_n,JC_m, JC_Tm, JC_Ts, \
#         damage_initiation_para,damage_evolution_para_1,damage_evolution_para_2,damage_max_degradation,Density]

#     fun_obj(x0)