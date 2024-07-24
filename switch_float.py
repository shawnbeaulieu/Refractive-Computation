import numpy as np

from ConfigPlot import ConfigPlot_EigenMode_DiffMass, ConfigPlot_YFixed_rec, ConfigPlot_DiffMass_SP
from MD_functions import MD_VibrSP_ConstV_Yfixed_DiffK, FIRE_YFixed_ConstV_DiffK, MD_VibrSP_ConstV_Yfixed_DiffK2
from DynamicalMatrix import DM_mass_DiffK_Yfixed
from plot_functions import Line_single, Line_multi
from ConfigPlot import ConfigPlot_DiffStiffness3, ConfigPlot_DiffStiffness4
import random
import matplotlib.pyplot as plt
import pickle
import itertools
from os.path import exists

class switch():

    def __init__(self, 
                 stiffness, 
                 ind1, 
                 ind2, 
                 out1,
                 source):

        # %% Initial Configuration
        # k1 = 1.
        # k2 = 10.

        self.stiffness = stiffness
        self.ind1 = ind1
        self.ind2 = ind2

        self.n_col = 6
        self.n_row = 5
        self.N = self.n_col * self.n_row

        Nt_fire = 1e6

        dt_ratio = 40
        Nt_SD = 1e5
        Nt_MD = 1e5

        dphi_index = -1
        dphi = 10 ** dphi_index
        d0 = 0.1
        d_ratio = 1.1
        self.Lx = d0 * self.n_col
        self.Ly = (self.n_row - 1) * np.sqrt(3) / 2 * d0 + d0

        x0 = np.zeros(self.N)
        y0 = np.zeros(self.N)

        phi0 = self.N * np.pi * d0 ** 2 / 4 / (self.Lx * self.Ly)
        d_ini = d0 * np.sqrt(1 + dphi / phi0)
        self.D = np.zeros(self.N) + d_ini
        # D = np.zeros(N)+d0

        self.x0 = np.zeros(self.N)
        self.y0 = np.zeros(self.N)

        for i_row in range(1, self.n_row + 1):
            for i_col in range(1, self.n_col + 1):
                ind = (i_row - 1) * self.n_col + i_col - 1
                if i_row % 2 == 1:
                    self.x0[ind] = (i_col - 1) * d0
                else:
                    self.x0[ind] = (i_col - 1) * d0 + 0.5 * d0
                self.y0[ind] = (i_row - 1) * np.sqrt(3) / 2 * d0
        self.y0 = self.y0 + 0.5 * d0

        self.mass = np.zeros(self.N) + 1
        # k_list = np.array([k1, k2, k1 * k2 / (k1 + k2)])
        # k_type = indices

        # Steepest Descent to get energy minimum
        self.x_ini, self.y_ini, self.p_now = \
            FIRE_YFixed_ConstV_DiffK(Nt_fire, 
                                     self.N, 
                                     self.x0, 
                                     self.y0, 
                                     self.D, 
                                     self.mass, 
                                     self.Lx, 
                                     self.Ly,
                                     self.stiffness)

        # specify the input ports and the output port
        self.ind_in1 = self.ind1  # int((n_col+1)/2) - 1 + n_col * 2
        self.ind_in2 = self.ind2  # ind_in1 + 2
        self.ind_out1 = out1#int(self.N - int((self.n_col + 1) / 2))
        self.ind_s = source#int((self.n_col + 1) / 2)
        # ind_fix = int((n_row+1)/2)*n_col-int((n_col+1)/2)

        self.B = 1
        self.Nt = 5e3  # it was 1e5 before, i reduced it to run faster
        self.Amp_Vibr = 1e-3
        self.Freq_Vibr1 = 12
        self.Freq_Vibr2 = 9
        self.Freq_Vibr3 = 18
        self.Freq_Vibr4 = 4
        # we are designing a nand gate at this amplitude and this freq

    def evaluate(self):

        rand_p_1 = ['10', '11', '01', '00']
        rand_p_2 = ['01', '00', '10', '11']
        rand_p_3 = ['00', '01', '11', '10']
        rand_p_4 = ['11', '10', '00', '01']

        gains_1 = {p: None for p in rand_p_1}
        gains_2 = {p: None for p in rand_p_2}
        gains_3 = {p: None for p in rand_p_3}
        gains_4 = {p: None for p in rand_p_4}

        nand_perms = ['00', '01', '10']
        and_perms = ['11']
        or_perms = ['01', '10', '11']
        nor_perms = ['00']

        nand_1 = []
        and_2 = []
        nor_3 = []
        or_4 = []

        for p in range(len(rand_p_1)):

            idx_1 = rand_p_1[p]
            idx_2 = rand_p_2[p]
            idx_3 = rand_p_3[p]
            idx_4 = rand_p_4[p]

            gains_1[idx_1], gains_2[idx_2], gains_3[idx_3], gains_4[idx_4] = \
                self.stimulate(int(idx_1[0]),
                               int(idx_1[1]),
                               int(idx_2[0]),
                               int(idx_2[1]),
                               int(idx_3[0]),
                               int(idx_3[1]),
                               int(idx_4[0]),
                               int(idx_4[1]))

            if idx_1 in nand_perms:
                nand_1.append((1.0 - np.clip(gains_1[idx_1], 0,1))**2)
            else:
                nand_1.append((0.0 - np.clip(gains_1[idx_1], 0,1))**2)

            if idx_2 in and_perms:
                and_2.append((1.0 - np.clip(gains_2[idx_2], 0,1))**2)
            else:
                and_2.append((0.0 - np.clip(gains_2[idx_2], 0,1))**2)

            if idx_3 in nor_perms:
                nor_3.append((1.0 - np.clip(gains_3[idx_3], 0,1))**2)
            else:
                nor_3.append((0.0 - np.clip(gains_3[idx_3], 0,1))**2)

            if idx_4 in or_perms:
                or_4.append((1.0 - np.clip(gains_4[idx_4], 0,1))**2)
            else:
                or_4.append((0.0 - np.clip(gains_4[idx_4], 0,1))**2)


        nandness = np.sum(nand_1)
        andness = np.sum(and_2)
        norness = np.sum(nor_3)
        orness = np.sum(or_4)

        return([nandness,
                andness,
                orness,
                norness])

    def stimulate(self, n1_f1, n2_f1, n1_f2, n2_f2, n1_f3, n2_f3, n1_f4, n2_f4):

        AV1_F1 = n1_f1 * self.Amp_Vibr
        AV2_F1 = n2_f1 * self.Amp_Vibr
        AV1_F2 = n1_f2 * self.Amp_Vibr
        AV2_F2 = n2_f2 * self.Amp_Vibr
        AV1_F3 = n1_f3 * self.Amp_Vibr
        AV2_F3 = n2_f3 * self.Amp_Vibr
        AV1_F4 = n1_f4 * self.Amp_Vibr
        AV2_F4 = n2_f4 * self.Amp_Vibr

        sim_outputs = MD_VibrSP_ConstV_Yfixed_DiffK(self.stiffness,
                                                    self.B,
                                                    self.Nt,
                                                    self.N,
                                                    self.x_ini,
                                                    self.y_ini,
                                                    self.D,
                                                    self.mass,
                                                    [self.Lx, self.Ly],
                                                    self.Freq_Vibr1,
                                                    self.Freq_Vibr2,
                                                    self.Freq_Vibr3,
                                                    self.Freq_Vibr4,
                                                    self.Amp_Vibr,
                                                    self.ind_s,
                                                    self.ind_in1,
                                                    self.ind_in2,
                                                    self.ind_out1,
                                                    AV1_F1,
                                                    AV2_F1,
                                                    AV1_F2,
                                                    AV2_F2,
                                                    AV1_F3,
                                                    AV2_F3,
                                                    AV1_F4,
                                                    AV2_F4)
        freq_fft = sim_outputs[0]
        fft_in1_f1 = sim_outputs[1]
        fft_in2_f1 = sim_outputs[2]
        fft_in1_f2 = sim_outputs[3]
        fft_in2_f2 = sim_outputs[4]
        fft_in1_f3 = sim_outputs[5]
        fft_in2_f3 = sim_outputs[6]
        fft_in1_f4 = sim_outputs[7]
        fft_in2_f4 = sim_outputs[8]
        fft_s_f1 = sim_outputs[9]
        fft_s_f2 = sim_outputs[10]
        fft_s_f3 = sim_outputs[11]
        fft_s_f4 = sim_outputs[12]
        fft_x_out1_f1 = sim_outputs[13]
        fft_y_out1_f1 = sim_outputs[14]
        fft_x_out1_f2 = sim_outputs[15]
        fft_y_out1_f2 = sim_outputs[16]
        fft_x_out1_f3 = sim_outputs[17]
        fft_y_out1_f3 = sim_outputs[18]
        fft_x_out1_f4 = sim_outputs[19]
        fft_y_out1_f4 = sim_outputs[20]
        mean_cont = sim_outputs[21]
        nt_rec = sim_outputs[22]
        Ek_now = sim_outputs[23]
        Ep_now = sim_outputs[24]
        cont_now = sim_outputs[25]

        idx1 = np.where(freq_fft > self.Freq_Vibr1)[0][0]
        idx2 = np.where(freq_fft > self.Freq_Vibr2)[0][0]
        idx3 = np.where(freq_fft > self.Freq_Vibr3)[0][0]
        idx4 = np.where(freq_fft > self.Freq_Vibr4)[0][0]

        # fft of the output port at the driving frequency

        out1_f1 = fft_x_out1_f1[idx1 - 1] + (fft_x_out1_f1[idx1] - fft_x_out1_f1[idx1 - 1]) * (
                    (self.Freq_Vibr1 - freq_fft[idx1 - 1]) / (freq_fft[idx1] - freq_fft[idx1 - 1]))
        out1_f2 = fft_x_out1_f2[idx2 - 1] + (fft_x_out1_f2[idx2] - fft_x_out1_f2[idx2 - 1]) * (
                    (self.Freq_Vibr2 - freq_fft[idx2 - 1]) / (freq_fft[idx2] - freq_fft[idx2 - 1]))

        out1_f3 = fft_x_out1_f3[idx3 - 1] + (fft_x_out1_f3[idx3] - fft_x_out1_f3[idx3 - 1]) * (
                    (self.Freq_Vibr3 - freq_fft[idx3 - 1]) / (freq_fft[idx3] - freq_fft[idx3 - 1]))
        out1_f4 = fft_x_out1_f4[idx4 - 1] + (fft_x_out1_f4[idx4] - fft_x_out1_f4[idx4 - 1]) * (
                    (self.Freq_Vibr4 - freq_fft[idx4 - 1]) / (freq_fft[idx4] - freq_fft[idx4 - 1]))

        # fft of input1 at driving frequency

        inp1_f1 = fft_in1_f1[idx1 - 1] + (fft_in1_f1[idx1] - fft_in1_f1[idx1 - 1]) * (
                    (self.Freq_Vibr1 - freq_fft[idx1 - 1]) / (freq_fft[idx1] - freq_fft[idx1 - 1]))
        inp2_f1 = fft_in2_f1[idx1 - 1] + (fft_in2_f1[idx1] - fft_in2_f1[idx1 - 1]) * (
                    (self.Freq_Vibr1 - freq_fft[idx1 - 1]) / (freq_fft[idx1] - freq_fft[idx1 - 1]))

        inp1_f2 = fft_in1_f2[idx2 - 1] + (fft_in1_f2[idx2] - fft_in1_f2[idx2 - 1]) * (
                    (self.Freq_Vibr2 - freq_fft[idx2 - 1]) / (freq_fft[idx2] - freq_fft[idx2 - 1]))
        inp2_f2 = fft_in2_f2[idx2 - 1] + (fft_in2_f2[idx2] - fft_in2_f2[idx2 - 1]) * (
                    (self.Freq_Vibr2 - freq_fft[idx2 - 1]) / (freq_fft[idx2] - freq_fft[idx2 - 1]))

        inp1_f3 = fft_in1_f3[idx3 - 1] + (fft_in1_f3[idx3] - fft_in1_f3[idx3 - 1]) * (
                    (self.Freq_Vibr3 - freq_fft[idx3 - 1]) / (freq_fft[idx3] - freq_fft[idx3 - 1]))
        inp2_f3 = fft_in2_f3[idx3 - 1] + (fft_in2_f3[idx3] - fft_in2_f3[idx3 - 1]) * (
                    (self.Freq_Vibr3 - freq_fft[idx3 - 1]) / (freq_fft[idx3] - freq_fft[idx3 - 1]))

        inp1_f4 = fft_in1_f4[idx4 - 1] + (fft_in1_f4[idx4] - fft_in1_f4[idx4 - 1]) * (
                    (self.Freq_Vibr4 - freq_fft[idx4 - 1]) / (freq_fft[idx4] - freq_fft[idx4 - 1]))
        inp2_f4 = fft_in2_f4[idx4 - 1] + (fft_in2_f4[idx4] - fft_in2_f4[idx4 - 1]) * (
                    (self.Freq_Vibr4 - freq_fft[idx4 - 1]) / (freq_fft[idx4] - freq_fft[idx4 - 1]))

        # fft of source at driving frequency

        src_f1 = fft_s_f1[idx1 - 1] + (fft_s_f1[idx1] - fft_s_f1[idx1 - 1]) * (
                    (self.Freq_Vibr1 - freq_fft[idx1 - 1]) / (freq_fft[idx1] - freq_fft[idx1 - 1]))

        src_f2 = fft_s_f2[idx2 - 1] + (fft_s_f2[idx2] - fft_s_f2[idx2 - 1]) * (
                    (self.Freq_Vibr2 - freq_fft[idx2 - 1]) / (freq_fft[idx2] - freq_fft[idx2 - 1]))

        src_f3 = fft_s_f3[idx3 - 1] + (fft_s_f3[idx3] - fft_s_f3[idx3 - 1]) * (
                    (self.Freq_Vibr3 - freq_fft[idx3 - 1]) / (freq_fft[idx3] - freq_fft[idx3 - 1]))

        src_f4 = fft_s_f4[idx4 - 1] + (fft_s_f4[idx4] - fft_s_f4[idx4 - 1]) * (
                    (self.Freq_Vibr4 - freq_fft[idx4 - 1]) / (freq_fft[idx4] - freq_fft[idx4 - 1]))

        gain1 = out1_f1 / (src_f1 + inp1_f1 + inp2_f1)
        gain2 = out1_f2 / (src_f2 + inp1_f2 + inp2_f2)
        gain3 = out1_f3 / (src_f3 + inp1_f3 + inp2_f3)
        gain4 = out1_f4 / (src_f4 + inp1_f4 + inp2_f4)

        return (gain1, gain2, gain3, gain4)


    def computeGateness(self, gate=1):

        rand_p_1 = ['10', '11', '01', '00']
        rand_p_2 = ['01', '00', '10', '11']
        rand_p_3 = ['00', '01', '11', '10']
        rand_p_4 = ['11', '10', '00', '01']

        nand_perms = ['00', '01', '10']
        and_perms = ['11']
        or_perms = ['01', '10', '11']
        nor_perms = ['00']

        gateness = {}

        for output_node in range(self.N):

            self.ind_out1 = output_node
            gateness[output_node] = []

            for new_freq in np.arange(1,40,1):

                self.Freq_Vibr4 = new_freq    

                nand_1 = []
                and_2 = []
                nor_3 = []
                or_4 = []

                gains_1 = {p: None for p in rand_p_1}
                gains_2 = {p: None for p in rand_p_2}
                gains_3 = {p: None for p in rand_p_3}
                gains_4 = {p: None for p in rand_p_4}

                for p in range(len(rand_p_1)):
  
                    idx_1 = rand_p_1[p]
                    idx_2 = rand_p_2[p]
                    idx_3 = rand_p_3[p]
                    idx_4 = rand_p_4[p]

                    gains_1[idx_1], gains_2[idx_2], gains_3[idx_3], gains_4[idx_4] = \
                        self.stimulate(int(idx_1[0]),
                                       int(idx_1[1]),
                                       int(idx_2[0]),
                                       int(idx_2[1]),
                                       int(idx_3[0]),
                                       int(idx_3[1]),
                                       int(idx_4[0]),
                                       int(idx_4[1]))

                    if idx_1 in nand_perms:
                        nand_1.append((1.0 - np.clip(gains_1[idx_1], 0,1))**2)
                    else:
                        nand_1.append((0.0 - np.clip(gains_1[idx_1], 0,1))**2)

                    if idx_2 in and_perms:
                        and_2.append((1.0 - np.clip(gains_2[idx_2], 0,1))**2)
                    else:
                        and_2.append((0.0 - np.clip(gains_2[idx_2], 0,1))**2)

                    if idx_3 in nor_perms:
                        nor_3.append((1.0 - np.clip(gains_3[idx_3], 0,1))**2)
                    else:
                        nor_3.append((0.0 - np.clip(gains_3[idx_3], 0,1))**2)

                    if idx_4 in or_perms:
                        or_4.append((1.0 - np.clip(gains_4[idx_4], 0,1))**2)
                    else:
                        or_4.append((0.0 - np.clip(gains_4[idx_4], 0,1))**2)


                nandness = np.sum(nand_1)
                andness = np.sum(and_2)
                norness = np.sum(nor_3)
                orness = np.sum(or_4)
         
                if gate == 1:
                    print(nandness)
                    gateness[output_node].append(nandness)
                elif gate == 2:
                    print(andness)
                    gateness[output_node].append(andness)
                elif gate == 3:
                    print(norness)
                    gateness[output_node].append(norness)
                elif gate == 4:
                    print(orness)
                    gateness[output_node].append(orness)

            np.save("outnode2freq_gateness_of_gate={0}".format(gate), gateness)


    def showPacking(stiffness, ind1, ind2):
        #k1 = 1.
        #k2 = 10.

        n_col = 6
        n_row = 5
        N = n_col*n_row

        m1=1
        m2=10
        
        dphi_index = -1
        dphi = 10**dphi_index
        d0 = 0.1
        Lx = d0*n_col
        Ly = (n_row-1)*np.sqrt(3)/2*d0+d0
        
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        
        phi0 = N*np.pi*d0**2/4/(Lx*Ly)
        d_ini = d0*np.sqrt(1+dphi/phi0)
        D = np.zeros(N)+d_ini
        #D = np.zeros(N)+d0 
        
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        for i_row in range(1, n_row+1):
            for i_col in range(1, n_col+1):
                ind = (i_row-1)*n_col+i_col-1
                if i_row%2 == 1:
                    x0[ind] = (i_col-1)*d0
                else:
                    x0[ind] = (i_col-1)*d0+0.5*d0
                y0[ind] = (i_row-1)*np.sqrt(3)/2*d0
        y0 = y0+0.5*d0
        
        mass = np.zeros(N) + 1
        #k_list = np.array([k1, k2, k1 * k2 / (k1 + k2)])
        #k_type = indices

        # specify the input ports and the output port
        ind_in1 = ind1 #int((n_col+1)/2) - 1 + n_col * 2
        ind_in2 = ind2 #ind_in1 + 2
        ind_out = int(N-int((n_col+1)/2))
        ind_s = int((n_col+1)/2)
        #ind_fix = int((n_row+1)/2)*n_col-int((n_col+1)/2)

        # show packing
        ConfigPlot_DiffStiffness4(N, x0, y0, D, [Lx,Ly], stiffness, 0, '/Users/atoosa/Desktop/results/packing.pdf', ind_in1, ind_in2, ind_s, ind_out)


    def plotInOut(stiffness, ind1, ind2):

        #%% Initial Configuration
        #k1 = 1.
        #k2 = 10. 
        m1 = 1
        m2 = 10
        
        n_col = 6
        n_row = 5
        N = n_col*n_row
        
        Nt_fire = 1e6
        
        dt_ratio = 40
        Nt_SD = 1e5
        Nt_MD = 1e5
        
        
        dphi_index = -1
        dphi = 10**dphi_index
        d0 = 0.1
        d_ratio = 1.1
        Lx = d0*n_col
        Ly = (n_row-1)*np.sqrt(3)/2*d0+d0
        
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        
        phi0 = N*np.pi*d0**2/4/(Lx*Ly)
        d_ini = d0*np.sqrt(1+dphi/phi0)
        D = np.zeros(N)+d_ini
        #D = np.zeros(N)+d0 
        
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        for i_row in range(1, n_row+1):
            for i_col in range(1, n_col+1):
                ind = (i_row-1)*n_col+i_col-1
                if i_row%2 == 1:
                    x0[ind] = (i_col-1)*d0
                else:
                    x0[ind] = (i_col-1)*d0+0.5*d0
                y0[ind] = (i_row-1)*np.sqrt(3)/2*d0
        y0 = y0+0.5*d0
        
        mass = np.zeros(N) + 1
        #k_list = np.array([k1, k2, k1 * k2 / (k1 + k2)])
        #k_type = indices
        
        # Steepest Descent to get energy minimum      
        x_ini, y_ini, p_now = FIRE_YFixed_ConstV_DiffK(Nt_fire, N, x0, y0, D, mass, Lx, Ly, stiffness)

        # calculating the bandgap - no need to do this in this problem
        w, v = DM_mass_DiffK_Yfixed(N, x_ini, y_ini, D, mass, Lx, 0.0, Ly, stiffness)
        w = np.real(w)
        v = np.real(v)
        freq = np.sqrt(np.absolute(w))
        ind_sort = np.argsort(freq)
        freq = freq[ind_sort]
        v = v[:, ind_sort]
        ind = freq > 1e-4
        eigen_freq = freq[ind]
        eigen_mode = v[:, ind]
        w_delta = eigen_freq[1:] - eigen_freq[0:-1]
        index = np.argmax(w_delta)
        F_low_exp = eigen_freq[index]
        F_high_exp = eigen_freq[index+1]

        plt.figure(figsize=(6.4,4.8))
        ax = plt.axes()
        plt.scatter(np.arange(0, len(eigen_freq)), eigen_freq, marker='x', color='blue', s=200, linewidth=3)
        plt.xlabel(r"Index $(k)$", fontsize=32)
        plt.ylabel(r"Frequency $(\omega)$", fontsize=32)
        plt.title("Frequency Spectrum", fontsize=32, fontweight="bold")
        plt.grid(color='skyblue', linestyle=':', linewidth=0.75)
        props = dict(facecolor='green', alpha=0.1)
        myText = r'$\omega_{low}=$'+"{:.2f}".format(F_low_exp)+"\n"+r'$\omega_{high}=$'+"{:.2f}".format(F_high_exp)+"\n"+r'$\Delta \omega=$'+"{:.2f}".format(max(w_delta))
        plt.text(0.78, 0.15, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=28, bbox=props)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.tight_layout()
        plt.show()

        print("specs:")

        print(F_low_exp)
        print(F_high_exp)
        print(max(w_delta))

        # specify the input ports and the output port
        ind_in1 = ind1 #int((n_col+1)/2) - 1 + n_col * 2
        ind_in2 = ind2 #ind_in1 + 2
        ind_out = int(N-int((n_col+1)/2))
        ind_s = int((n_col+1)/2)
        #ind_fix = int((n_row+1)/2)*n_col-int((n_col+1)/2)

        B = 1
        Nt = 1e4 # it was 1e5 before, i reduced it to run faster

        # we are designing a nand gate at this amplitude and this freq
        Amp_Vibr = 1e-3
        Freq_Vibr = 10

        # case 00: output: 1
        Amp_Vibr1 = 0 * Amp_Vibr
        Amp_Vibr2 = 0 * Amp_Vibr
        
        # changed the resonator to one in MD_functions file and vibrations in x direction
        freq_fft, fft_in1, fft_in2, fft_s, fft_x_out, fft_y_out, mean_cont, nt_rec, Ek_now, Ep_now, cont_now = MD_VibrSP_ConstV_Yfixed_DiffK(stiffness, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr, ind_s, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

        ind = np.where(freq_fft>Freq_Vibr)
        index_=ind[0][0]
        # fft of the output port at the driving frequency
        out1 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))
        # fft of input1 at driving frequency
        inp1 = fft_in1[index_-1] + (fft_in1[index_]-fft_in1[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

        ind = np.where(freq_fft>Freq_Vibr)
        index_=ind[0][0]
        # fft of the output port at the driving frequency
        out2 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))
        # fft of input2 at driving frequency
        inp2 = fft_in2[index_-1] + (fft_in2[index_]-fft_in2[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

        # fft of source at driving frequency
        src = fft_s[index_-1] + (fft_s[index_]-fft_s[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

        gain00 = out1/(src+inp1+inp2)

        fig = plt.figure(figsize=(6.4,4.8))
        ax = plt.axes()
        plt.plot(freq_fft, fft_in1, color='lightgreen', label='Input1', linestyle='solid', linewidth=5)
        plt.plot(freq_fft, fft_in2, color='blue', label='Input2', linestyle='dashed', linewidth=1)
        plt.plot(freq_fft, fft_x_out, color='red', label='Output', linestyle='solid', linewidth=2)
        plt.xlabel("Frequency", fontsize=18)
        plt.ylabel("Amplitude of FFT", fontsize=18)
        plt.title("input = 00", fontsize=18, fontweight="bold")
        plt.grid(color='skyblue', linestyle=':', linewidth=0.75)
        plt.legend(loc='upper right', fontsize=16)
        #plt.axvline(x=Freq_Vibr, color='purple', linestyle='solid', alpha=0.5)
        myText = 'Gain='+"{:.4f}".format(gain00)
        plt.text(0.5, 0.9, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=18)
        plt.ylim((0, 0.0015))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.show()

        x_in1, x_in2, x_s, x_out = MD_VibrSP_ConstV_Yfixed_DiffK2(stiffness, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr, ind_s, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

        fig = plt.figure(figsize=(6.4,4.8))
        ax = plt.axes()
        plt.plot(x_in1, color='lightgreen', label='Input1', linestyle='solid', linewidth=5)
        plt.plot(x_in2, color='blue', label='Input2', linestyle='dashed', linewidth=1)
        plt.plot(x_out-np.mean(x_out[1000:]), color='red', label='Output', linestyle='solid', linewidth=2)
        plt.hlines(y=0.001, xmin=0, xmax=1e4, linewidth=2, linestyle='--', color='magenta', alpha=0.8)
        plt.hlines(y=-0.001, xmin=0, xmax=1e4, linewidth=2, linestyle='--', color='magenta', alpha=0.8)
        plt.xlabel("Time Steps", fontsize=18)
        plt.ylabel("Displacement", fontsize=18)
        plt.title("input = 00", fontsize=18, fontweight="bold")
        plt.grid(color='skyblue', linestyle=':', linewidth=0.75)
        plt.legend(loc='upper right', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim((-0.0025, 0.0025))
        plt.tight_layout()
        plt.show()

        # case 01: output: 1
        Amp_Vibr1 = 0 * Amp_Vibr
        Amp_Vibr2 = 1 * Amp_Vibr
        
        # changed the resonator to one in MD_functions file and vibrations in x direction
        freq_fft, fft_in1, fft_in2, fft_s, fft_x_out, fft_y_out, mean_cont, nt_rec, Ek_now, Ep_now, cont_now = MD_VibrSP_ConstV_Yfixed_DiffK(stiffness, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr, ind_s, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

        ind = np.where(freq_fft>Freq_Vibr)
        index_=ind[0][0]
        # fft of the output port at the driving frequency
        out1 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))
        # fft of input1 at driving frequency
        inp1 = fft_in1[index_-1] + (fft_in1[index_]-fft_in1[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

        ind = np.where(freq_fft>Freq_Vibr)
        index_=ind[0][0]
        # fft of the output port at the driving frequency
        out2 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))
        # fft of input2 at driving frequency
        inp2 = fft_in2[index_-1] + (fft_in2[index_]-fft_in2[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

        # fft of source at driving frequency
        src = fft_s[index_-1] + (fft_s[index_]-fft_s[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

        gain01 = out1/((src+inp1+inp2))

        fig = plt.figure(figsize=(6.4,4.8))
        ax = plt.axes()
        plt.plot(freq_fft, fft_in1, color='lightgreen', label='Input1', linestyle='solid', linewidth=5)
        plt.plot(freq_fft, fft_in2, color='blue', label='Input2', linestyle='dashed', linewidth=1)
        plt.plot(freq_fft, fft_x_out, color='red', label='Output', linestyle='solid', linewidth=2)
        plt.xlabel("Frequency", fontsize=18)
        plt.ylabel("Amplitude of FFT", fontsize=18)
        plt.title("input = 01", fontsize=18, fontweight="bold")
        plt.grid(color='skyblue', linestyle=':', linewidth=0.75)
        #plt.legend(loc='upper right', fontsize=16)
        #plt.axvline(x=Freq_Vibr, color='purple', linestyle='solid', alpha=0.5)
        myText = 'Gain='+"{:.4f}".format(gain01)
        plt.text(0.5, 0.9, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim((0, 0.0015))
        plt.tight_layout()
        plt.show()

        x_in1, x_in2, x_s, x_out = MD_VibrSP_ConstV_Yfixed_DiffK2(stiffness, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr, ind_s, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

        fig = plt.figure(figsize=(6.4,4.8))
        ax = plt.axes()
        plt.plot(x_in1, color='lightgreen', label='Input1', linestyle='solid', linewidth=5)
        plt.plot(x_in2, color='blue', label='Input2', linestyle='dashed', linewidth=1)
        plt.plot(x_out-np.mean(x_out[1000:]), color='red', label='Output', linestyle='solid', linewidth=2)
        plt.hlines(y=0.001, xmin=0, xmax=1e4, linewidth=2, linestyle='--', color='magenta', alpha=0.8)
        plt.hlines(y=-0.001, xmin=0, xmax=1e4, linewidth=2, linestyle='--', color='magenta', alpha=0.8)
        plt.xlabel("Time Steps", fontsize=18)
        plt.ylabel("Displacement", fontsize=18)
        plt.title("input = 01", fontsize=18, fontweight="bold")
        plt.grid(color='skyblue', linestyle=':', linewidth=0.75)
        #plt.legend(loc='upper right', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim((-0.0025, 0.0025))
        plt.tight_layout()
        plt.show()

        # case 10: output: 1
        Amp_Vibr1 = 1 * Amp_Vibr
        Amp_Vibr2 = 0 * Amp_Vibr
        
        # changed the resonator to one in MD_functions file and vibrations in x direction
        freq_fft, fft_in1, fft_in2, fft_s, fft_x_out, fft_y_out, mean_cont, nt_rec, Ek_now, Ep_now, cont_now = MD_VibrSP_ConstV_Yfixed_DiffK(stiffness, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr, ind_s, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

        ind = np.where(freq_fft>Freq_Vibr)
        index_=ind[0][0]
        # fft of the output port at the driving frequency
        out1 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))
        # fft of input1 at driving frequency
        inp1 = fft_in1[index_-1] + (fft_in1[index_]-fft_in1[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

        ind = np.where(freq_fft>Freq_Vibr)
        index_=ind[0][0]
        # fft of the output port at the driving frequency
        out2 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))
        # fft of input2 at driving frequency
        inp2 = fft_in2[index_-1] + (fft_in2[index_]-fft_in2[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

        # fft of source at driving frequency
        src = fft_s[index_-1] + (fft_s[index_]-fft_s[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

        gain10 = out1/((src+inp1+inp2))

        fig = plt.figure(figsize=(6.4,4.8))
        ax = plt.axes()
        plt.plot(freq_fft, fft_in1, color='lightgreen', label='Input1', linestyle='solid', linewidth=5)
        plt.plot(freq_fft, fft_in2, color='blue', label='Input2', linestyle='dashed', linewidth=1)
        plt.plot(freq_fft, fft_x_out, color='red', label='Output', linestyle='solid', linewidth=2)
        plt.xlabel("Frequency", fontsize=18)
        plt.ylabel("Amplitude of FFT", fontsize=18)
        plt.title("input = 10", fontsize=18, fontweight="bold")
        plt.grid(color='skyblue', linestyle=':', linewidth=0.75)
        #plt.legend(loc='upper right', fontsize=16)
        #plt.axvline(x=Freq_Vibr, color='purple', linestyle='solid', alpha=0.5)
        myText = 'Gain='+"{:.4f}".format(gain10)
        plt.text(0.5, 0.9, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim((0, 0.0015))
        plt.tight_layout()
        plt.show()

        x_in1, x_in2, x_s, x_out = MD_VibrSP_ConstV_Yfixed_DiffK2(stiffness, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr, ind_s, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)
        
        fig = plt.figure(figsize=(6.4,4.8))
        ax = plt.axes()
        plt.plot(x_in1, color='lightgreen', label='Input1', linestyle='solid', linewidth=5)
        plt.plot(x_in2, color='blue', label='Input2', linestyle='dashed', linewidth=1)
        plt.plot(x_out-np.mean(x_out[1000:]), color='red', label='Output', linestyle='solid', linewidth=2)
        plt.hlines(y=0.001, xmin=0, xmax=1e4, linewidth=2, linestyle='--', color='magenta', alpha=0.8)
        plt.hlines(y=-0.001, xmin=0, xmax=1e4, linewidth=2, linestyle='--', color='magenta', alpha=0.8)
        plt.xlabel("Time Steps", fontsize=18)
        plt.ylabel("Displacement", fontsize=18)
        plt.title("input = 10", fontsize=18, fontweight="bold")
        plt.grid(color='skyblue', linestyle=':', linewidth=0.75)
        #plt.legend(loc='upper right', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim((-0.0025, 0.0025))
        plt.tight_layout()
        plt.show()

        # case 11: output: 0
        Amp_Vibr1 = 1 * Amp_Vibr
        Amp_Vibr2 = 1 * Amp_Vibr
        
        # changed the resonator to one in MD_functions file and vibrations in x direction
        freq_fft, fft_in1, fft_in2, fft_s, fft_x_out, fft_y_out, mean_cont, nt_rec, Ek_now, Ep_now, cont_now = MD_VibrSP_ConstV_Yfixed_DiffK(stiffness, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr, ind_s, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

        ind = np.where(freq_fft>Freq_Vibr)
        index_=ind[0][0]
        # fft of the output port at the driving frequency
        out1 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))
        # fft of input1 at driving frequency
        inp1 = fft_in1[index_-1] + (fft_in1[index_]-fft_in1[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

        ind = np.where(freq_fft>Freq_Vibr)
        index_=ind[0][0]
        # fft of the output port at the driving frequency
        out2 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))
        # fft of input2 at driving frequency
        inp2 = fft_in2[index_-1] + (fft_in2[index_]-fft_in2[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

        # fft of source at driving frequency
        src = fft_s[index_-1] + (fft_s[index_]-fft_s[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

        gain11 = out1/((src+inp1+inp2))

        fig = plt.figure(figsize=(6.4,4.8))
        ax = plt.axes()
        plt.plot(freq_fft, fft_in1, color='lightgreen', label='Input1', linestyle='solid', linewidth=5)
        plt.plot(freq_fft, fft_in2, color='blue', label='Input2', linestyle='dashed', linewidth=1)
        plt.plot(freq_fft, fft_x_out, color='red', label='Output', linestyle='solid', linewidth=2)
        plt.xlabel("Frequency", fontsize=18)
        plt.ylabel("Amplitude of FFT", fontsize=18)
        plt.title("input = 11", fontsize=18, fontweight="bold")
        plt.grid(color='skyblue', linestyle=':', linewidth=0.75)
        #plt.legend(loc='upper right', fontsize=16)
        #plt.axvline(x=Freq_Vibr, color='purple', linestyle='solid', alpha=0.5)
        myText = 'Gain='+"{:.4f}".format(gain11)
        plt.text(0.5, 0.9, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim((0, 0.0015))
        plt.tight_layout()
        plt.show()

        x_in1, x_in2, x_s, x_out = MD_VibrSP_ConstV_Yfixed_DiffK2(stiffness, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr, ind_s, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

        fig = plt.figure(figsize=(6.4,4.8))
        ax = plt.axes()
        plt.plot(x_in1, color='lightgreen', label='Input1', linestyle='solid', linewidth=5)
        plt.plot(x_in2, color='blue', label='Input2', linestyle='dashed', linewidth=1)
        plt.plot(x_out-np.mean(x_out[1000:], axis=0), color='red', label='Output', linestyle='solid', linewidth=2)
        plt.hlines(y=0.001, xmin=0, xmax=1e4, linewidth=2, linestyle='--', color='magenta', alpha=0.8)
        plt.hlines(y=-0.001, xmin=0, xmax=1e4, linewidth=2, linestyle='--', color='magenta', alpha=0.8)
        plt.xlabel("Time Steps", fontsize=18)
        plt.ylabel("Displacement", fontsize=18)
        plt.title("input = 11", fontsize=18, fontweight="bold")
        plt.grid(color='skyblue', linestyle=':', linewidth=0.75)
        #plt.legend(loc='upper right', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim((-0.0025, 0.0025))
        plt.tight_layout()
        plt.show()

        print("gain00:")
        print(gain00)
        print("gain01:")
        print(gain01)
        print("gain10:")
        print(gain10)
        print("gain11:")
        print(gain11)

        nandness = (abs(1-round(gain00, 2)) + abs(1-round(gain01, 2)) + abs(1-round(gain10, 2)) + abs(0-round(gain11, 2)))/4
        print("nandness: "+str(round(nandness, 3)))



        # we are designing a nand gate at this amplitude and this freq
        Amp_Vibr = 1e-3
        Freq_Vibr = 20

        # case 00: output: 1
        Amp_Vibr1 = 0 * Amp_Vibr
        Amp_Vibr2 = 0 * Amp_Vibr
        
        # changed the resonator to one in MD_functions file and vibrations in x direction
        freq_fft, fft_in1, fft_in2, fft_s, fft_x_out, fft_y_out, mean_cont, nt_rec, Ek_now, Ep_now, cont_now = MD_VibrSP_ConstV_Yfixed_DiffK(stiffness, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr, ind_s, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

        ind = np.where(freq_fft>Freq_Vibr)
        index_=ind[0][0]
        # fft of the output port at the driving frequency
        out1 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))
        # fft of input1 at driving frequency
        inp1 = fft_in1[index_-1] + (fft_in1[index_]-fft_in1[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

        ind = np.where(freq_fft>Freq_Vibr)
        index_=ind[0][0]
        # fft of the output port at the driving frequency
        out2 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))
        # fft of input2 at driving frequency
        inp2 = fft_in2[index_-1] + (fft_in2[index_]-fft_in2[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

        # fft of source at driving frequency
        src = fft_s[index_-1] + (fft_s[index_]-fft_s[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

        gain00 = out1/(src+inp1+inp2)

        fig = plt.figure(figsize=(6.4,4.8))
        ax = plt.axes()
        plt.plot(freq_fft, fft_in1, color='lightgreen', label='Input1', linestyle='solid', linewidth=5)
        plt.plot(freq_fft, fft_in2, color='blue', label='Input2', linestyle='dashed', linewidth=1)
        plt.plot(freq_fft, fft_x_out, color='red', label='Output', linestyle='solid', linewidth=2)
        plt.xlabel("Frequency", fontsize=18)
        plt.ylabel("Amplitude of FFT", fontsize=18)
        plt.title("input = 00", fontsize=18, fontweight="bold")
        plt.grid(color='skyblue', linestyle=':', linewidth=0.75)
        plt.legend(loc='upper right', fontsize=16)
        #plt.axvline(x=Freq_Vibr, color='purple', linestyle='solid', alpha=0.5)
        myText = 'Gain='+"{:.4f}".format(gain00)
        plt.text(0.5, 0.9, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=18)
        plt.ylim((0, 0.0015))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.show()

        x_in1, x_in2, x_s, x_out = MD_VibrSP_ConstV_Yfixed_DiffK2(stiffness, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr, ind_s, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

        fig = plt.figure(figsize=(6.4,4.8))
        ax = plt.axes()
        plt.plot(x_in1, color='lightgreen', label='Input1', linestyle='solid', linewidth=5)
        plt.plot(x_in2, color='blue', label='Input2', linestyle='dashed', linewidth=1)
        plt.plot(x_out-np.mean(x_out[1000:]), color='red', label='Output', linestyle='solid', linewidth=2)
        plt.hlines(y=0.001, xmin=0, xmax=1e4, linewidth=2, linestyle='--', color='magenta', alpha=0.8)
        plt.hlines(y=-0.001, xmin=0, xmax=1e4, linewidth=2, linestyle='--', color='magenta', alpha=0.8)
        plt.xlabel("Time Steps", fontsize=18)
        plt.ylabel("Displacement", fontsize=18)
        plt.title("input = 00", fontsize=18, fontweight="bold")
        plt.grid(color='skyblue', linestyle=':', linewidth=0.75)
        plt.legend(loc='upper right', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim((-0.0025, 0.0025))
        plt.tight_layout()
        plt.show()

        # case 01: output: 1
        Amp_Vibr1 = 0 * Amp_Vibr
        Amp_Vibr2 = 1 * Amp_Vibr
        
        # changed the resonator to one in MD_functions file and vibrations in x direction
        freq_fft, fft_in1, fft_in2, fft_s, fft_x_out, fft_y_out, mean_cont, nt_rec, Ek_now, Ep_now, cont_now = MD_VibrSP_ConstV_Yfixed_DiffK(stiffness, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr, ind_s, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

        ind = np.where(freq_fft>Freq_Vibr)
        index_=ind[0][0]
        # fft of the output port at the driving frequency
        out1 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))
        # fft of input1 at driving frequency
        inp1 = fft_in1[index_-1] + (fft_in1[index_]-fft_in1[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

        ind = np.where(freq_fft>Freq_Vibr)
        index_=ind[0][0]
        # fft of the output port at the driving frequency
        out2 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))
        # fft of input2 at driving frequency
        inp2 = fft_in2[index_-1] + (fft_in2[index_]-fft_in2[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

        # fft of source at driving frequency
        src = fft_s[index_-1] + (fft_s[index_]-fft_s[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

        gain01 = out1/((src+inp1+inp2))

        fig = plt.figure(figsize=(6.4,4.8))
        ax = plt.axes()
        plt.plot(freq_fft, fft_in1, color='lightgreen', label='Input1', linestyle='solid', linewidth=5)
        plt.plot(freq_fft, fft_in2, color='blue', label='Input2', linestyle='dashed', linewidth=1)
        plt.plot(freq_fft, fft_x_out, color='red', label='Output', linestyle='solid', linewidth=2)
        plt.xlabel("Frequency", fontsize=18)
        plt.ylabel("Amplitude of FFT", fontsize=18)
        plt.title("input = 01", fontsize=18, fontweight="bold")
        plt.grid(color='skyblue', linestyle=':', linewidth=0.75)
        #plt.legend(loc='upper right', fontsize=16)
        #plt.axvline(x=Freq_Vibr, color='purple', linestyle='solid', alpha=0.5)
        myText = 'Gain='+"{:.4f}".format(gain01)
        plt.text(0.5, 0.9, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim((0, 0.0015))
        plt.tight_layout()
        plt.show()

        x_in1, x_in2, x_s, x_out = MD_VibrSP_ConstV_Yfixed_DiffK2(stiffness, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr, ind_s, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

        fig = plt.figure(figsize=(6.4,4.8))
        ax = plt.axes()
        plt.plot(x_in1, color='lightgreen', label='Input1', linestyle='solid', linewidth=5)
        plt.plot(x_in2, color='blue', label='Input2', linestyle='dashed', linewidth=1)
        plt.plot(x_out-np.mean(x_out[1000:]), color='red', label='Output', linestyle='solid', linewidth=2)
        plt.hlines(y=0.001, xmin=0, xmax=1e4, linewidth=2, linestyle='--', color='magenta', alpha=0.8)
        plt.hlines(y=-0.001, xmin=0, xmax=1e4, linewidth=2, linestyle='--', color='magenta', alpha=0.8)
        plt.xlabel("Time Steps", fontsize=18)
        plt.ylabel("Displacement", fontsize=18)
        plt.title("input = 01", fontsize=18, fontweight="bold")
        plt.grid(color='skyblue', linestyle=':', linewidth=0.75)
        #plt.legend(loc='upper right', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim((-0.0025, 0.0025))
        plt.tight_layout()
        plt.show()

        # case 10: output: 1
        Amp_Vibr1 = 1 * Amp_Vibr
        Amp_Vibr2 = 0 * Amp_Vibr
        
        # changed the resonator to one in MD_functions file and vibrations in x direction
        freq_fft, fft_in1, fft_in2, fft_s, fft_x_out, fft_y_out, mean_cont, nt_rec, Ek_now, Ep_now, cont_now = MD_VibrSP_ConstV_Yfixed_DiffK(stiffness, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr, ind_s, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

        ind = np.where(freq_fft>Freq_Vibr)
        index_=ind[0][0]
        # fft of the output port at the driving frequency
        out1 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))
        # fft of input1 at driving frequency
        inp1 = fft_in1[index_-1] + (fft_in1[index_]-fft_in1[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

        ind = np.where(freq_fft>Freq_Vibr)
        index_=ind[0][0]
        # fft of the output port at the driving frequency
        out2 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))
        # fft of input2 at driving frequency
        inp2 = fft_in2[index_-1] + (fft_in2[index_]-fft_in2[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

        # fft of source at driving frequency
        src = fft_s[index_-1] + (fft_s[index_]-fft_s[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

        gain10 = out1/((src+inp1+inp2))

        fig = plt.figure(figsize=(6.4,4.8))
        ax = plt.axes()
        plt.plot(freq_fft, fft_in1, color='lightgreen', label='Input1', linestyle='solid', linewidth=5)
        plt.plot(freq_fft, fft_in2, color='blue', label='Input2', linestyle='dashed', linewidth=1)
        plt.plot(freq_fft, fft_x_out, color='red', label='Output', linestyle='solid', linewidth=2)
        plt.xlabel("Frequency", fontsize=18)
        plt.ylabel("Amplitude of FFT", fontsize=18)
        plt.title("input = 10", fontsize=18, fontweight="bold")
        plt.grid(color='skyblue', linestyle=':', linewidth=0.75)
        #plt.legend(loc='upper right', fontsize=16)
        #plt.axvline(x=Freq_Vibr, color='purple', linestyle='solid', alpha=0.5)
        myText = 'Gain='+"{:.4f}".format(gain10)
        plt.text(0.5, 0.9, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim((0, 0.0015))
        plt.tight_layout()
        plt.show()

        x_in1, x_in2, x_s, x_out = MD_VibrSP_ConstV_Yfixed_DiffK2(stiffness, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr, ind_s, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)
        
        fig = plt.figure(figsize=(6.4,4.8))
        ax = plt.axes()
        plt.plot(x_in1, color='lightgreen', label='Input1', linestyle='solid', linewidth=5)
        plt.plot(x_in2, color='blue', label='Input2', linestyle='dashed', linewidth=1)
        plt.plot(x_out-np.mean(x_out[1000:]), color='red', label='Output', linestyle='solid', linewidth=2)
        plt.hlines(y=0.001, xmin=0, xmax=1e4, linewidth=2, linestyle='--', color='magenta', alpha=0.8)
        plt.hlines(y=-0.001, xmin=0, xmax=1e4, linewidth=2, linestyle='--', color='magenta', alpha=0.8)
        plt.xlabel("Time Steps", fontsize=18)
        plt.ylabel("Displacement", fontsize=18)
        plt.title("input = 10", fontsize=18, fontweight="bold")
        plt.grid(color='skyblue', linestyle=':', linewidth=0.75)
        #plt.legend(loc='upper right', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim((-0.0025, 0.0025))
        plt.tight_layout()
        plt.show()

        # case 11: output: 0
        Amp_Vibr1 = 1 * Amp_Vibr
        Amp_Vibr2 = 1 * Amp_Vibr
        
        # changed the resonator to one in MD_functions file and vibrations in x direction
        freq_fft, fft_in1, fft_in2, fft_s, fft_x_out, fft_y_out, mean_cont, nt_rec, Ek_now, Ep_now, cont_now = MD_VibrSP_ConstV_Yfixed_DiffK(stiffness, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr, ind_s, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

        ind = np.where(freq_fft>Freq_Vibr)
        index_=ind[0][0]
        # fft of the output port at the driving frequency
        out1 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))
        # fft of input1 at driving frequency
        inp1 = fft_in1[index_-1] + (fft_in1[index_]-fft_in1[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

        ind = np.where(freq_fft>Freq_Vibr)
        index_=ind[0][0]
        # fft of the output port at the driving frequency
        out2 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))
        # fft of input2 at driving frequency
        inp2 = fft_in2[index_-1] + (fft_in2[index_]-fft_in2[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

        # fft of source at driving frequency
        src = fft_s[index_-1] + (fft_s[index_]-fft_s[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

        gain11 = out1/((src+inp1+inp2))

        fig = plt.figure(figsize=(6.4,4.8))
        ax = plt.axes()
        plt.plot(freq_fft, fft_in1, color='lightgreen', label='Input1', linestyle='solid', linewidth=5)
        plt.plot(freq_fft, fft_in2, color='blue', label='Input2', linestyle='dashed', linewidth=1)
        plt.plot(freq_fft, fft_x_out, color='red', label='Output', linestyle='solid', linewidth=2)
        plt.xlabel("Frequency", fontsize=18)
        plt.ylabel("Amplitude of FFT", fontsize=18)
        plt.title("input = 11", fontsize=18, fontweight="bold")
        plt.grid(color='skyblue', linestyle=':', linewidth=0.75)
        #plt.legend(loc='upper right', fontsize=16)
        #plt.axvline(x=Freq_Vibr, color='purple', linestyle='solid', alpha=0.5)
        myText = 'Gain='+"{:.4f}".format(gain11)
        plt.text(0.5, 0.9, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim((0, 0.0015))
        plt.tight_layout()
        plt.show()

        x_in1, x_in2, x_s, x_out = MD_VibrSP_ConstV_Yfixed_DiffK2(stiffness, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr, ind_s, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

        fig = plt.figure(figsize=(6.4,4.8))
        ax = plt.axes()
        plt.plot(x_in1, color='lightgreen', label='Input1', linestyle='solid', linewidth=5)
        plt.plot(x_in2, color='blue', label='Input2', linestyle='dashed', linewidth=1)
        plt.plot(x_out-np.mean(x_out[1000:], axis=0), color='red', label='Output', linestyle='solid', linewidth=2)
        plt.hlines(y=0.001, xmin=0, xmax=1e4, linewidth=2, linestyle='--', color='magenta', alpha=0.8)
        plt.hlines(y=-0.001, xmin=0, xmax=1e4, linewidth=2, linestyle='--', color='magenta', alpha=0.8)
        plt.xlabel("Time Steps", fontsize=18)
        plt.ylabel("Displacement", fontsize=18)
        plt.title("input = 11", fontsize=18, fontweight="bold")
        plt.grid(color='skyblue', linestyle=':', linewidth=0.75)
        #plt.legend(loc='upper right', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim((-0.0025, 0.0025))
        plt.tight_layout()
        plt.show()

        print("gain00:")
        print(gain00)
        print("gain01:")
        print(gain01)
        print("gain10:")
        print(gain10)
        print("gain11:")
        print(gain11)

        nandness = (abs(1-round(gain00, 2)) + abs(1-round(gain01, 2)) + abs(1-round(gain10, 2)) + abs(0-round(gain11, 2)))/4
        print("nandness: "+str(round(nandness, 3)))

        return [gain00, gain01, gain10, gain11]

