import numpy as np
from dolfinx.fem import Function, FunctionSpace, assemble_scalar, form, Constant, locate_dofs_topological, dirichletbc, Expression
from dolfinx.fem.petsc import LinearProblem, assemble_matrix, assemble_vector
from dolfinx.la import matrix_csr
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square, create_interval, create_rectangle, locate_entities, meshtags
from dolfinx.geometry import BoundingBoxTree, compute_collisions, compute_colliding_cells
from ufl import dx, grad, nabla_grad, inner, dot, Measure, lhs, rhs
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import plot
import pyvista
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from exampleHelmholtz import Selftest
#from source.exampleHelmholtz import Selftest
plt.rcParams.update({
	"pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Helvetica"],
    'text.latex.preamble': r'\usepackage{amsfonts}'})
funcs = Selftest()

# auskommentiere Zeilen löschen?
class ConvergenceTester:
    """class which stores all necessary methods for convergence testing"""

    def __init__(self):
        number_conv_steps = 10
        start_value = 3
        end_value = 20
        #self.par_space = np.arange(start_value,end_value)
        self.par_space = np.linspace(start_value,end_value,number_conv_steps,dtype=int)
        self.number_mc_samples = 1
        self.high_res_ref = None
        self.y_points = None
        self.y_values_list = None
        self.n_sens = 11


    def compute_highres_reference(self,freq):
        """doc"""
        funcs.reset(ne=100)
        funcs.par1 = freq
        #funcs.snapshotsHandling()
        funcs.getFullOrderPrior()
        print("HIGH RES PRIOR DONE")
        
        #funcs.generateFakeData()
        #funcs.getFullOrderPosterior()
        self.high_res_ref = np.copy(funcs.u_mean_std)
        self.high_res_ref_fenics = Function(funcs.RBmodel.V)
        self.high_res_ref_fenics.x.array[:] = funcs.u_mean_std
        self.high_res_coords = np.copy(funcs.RBmodel.coordinates.tolist())
        

    def get_noisy_data_from_solution(self,n_sens,solution):
        """doc"""
        #idx = np.round(np.linspace(100, len(solution)-600, n_sens)).astype(int)
        #idx = np.round(np.linspace(0, 1000, n_sens)).astype(int)
        #idx = np.array([100,200,300,400,500])
        #idx = np.array([0,100,200,300,400,500,600,700,800,900,1000])
        idx = np.array([0,10,20,30,40,50,60,70,80,90,100])
        #self.y_points = [funcs.RBmodel.coordinates.tolist()[i] for i in idx]
        self.y_points = [self.high_res_coords[i] for i in idx]
        funcs.y_points = self.y_points
        #noise = 1.0e-16#2.5e-5
        noise = 2.5e-6
        values_at_indices = [solution[x]-0.0 for x in idx]
        n_obs = 100
        funcs.RBmodel.no = n_obs
        y_values_list = []
        for i in range(n_obs):
            y_values_list.append([x+np.random.normal(0,noise) for x in values_at_indices])
        a = np.array(y_values_list)
        self.y_values_list = a.tolist()
        funcs.y_values_list = self.y_values_list
        #print("points and values:")
        #print(self.y_points)
        #print(self.y_values_list)

    def computeErrorNorm(self,solution,reference):
       # norm = np.linalg.norm(np.abs(reference - solution))
        #norm = np.mean(solution)
       # error = reference - solution
        # bar_p_sq = 0
        # for p in solution:
        #     val = np.sqrt(np.abs(p*p))
        #     bar_p_sq+=val
        # bar_p_sq = bar_p_sq/np.shape(solution)[0]
        # mean_p = bar_p_sq

        # degree_raise = 0
        # uh = Function(funcs.RBmodel.V)
        # uh.x.array[:] = solution
        # if np.shape(reference)[0]==1001:
        #     u_ex = Function(funcs.RBmodel.V_ground_truth)
        # else:
        #     u_ex = Function(funcs.RBmodel.V)
        # u_ex.x.array[:] = reference
        # # Create higher order function space
        # degree = u_ex.function_space.ufl_element().degree()
        # family = u_ex.function_space.ufl_element().family()
        # mesh = u_ex.function_space.mesh
        # W = FunctionSpace(mesh, (family, degree+degree_raise))
        # # Interpolate approximate solution
        # u_W = Function(W)
        # u_W.interpolate(uh)

        # # Interpolate exact solution, special handling if exact solution
        # # is a ufl expression or a python lambda function
        # u_ex_W = Function(W)
        # u_ex_W.interpolate(u_ex)
        
        # # Compute the error in the higher order function space
        # e_W = Function(W)
        # e_W.x.array[:] = (np.real(u_W.x.array) - np.real(u_ex_W.x.array))#/np.real(u_ex_W.x.array)
        
        # # Integrate the error
        # error = form(inner(e_W, e_W) * ufl.dx)
        # error_H10 = form(dot(grad(e_W), grad(e_W)) * ufl.dx)
        # error_local = assemble_scalar(error)
        # error_local_H10 = assemble_scalar(error_H10)
        # error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
        # error_global_H10 = mesh.comm.allreduce(error_local_H10, op=MPI.SUM)

        # k = 2 * np.pi * funcs.par1 / 343
        # error_sobolev = np.sqrt(error_global) + k**(-2)*np.sqrt(error_global_H10)
        # error_sobolev = error_global# + k**(-2)*error_global_H10

        # u = form(inner(u_ex_W, u_ex_W) * ufl.dx)
        # u_H10 = form(dot(grad(u_ex_W), grad(u_ex_W)) * ufl.dx)
        # u_local = assemble_scalar(u)
        # u_local_H10 = assemble_scalar(u_H10)
        # u_global = mesh.comm.allreduce(u_local, op=MPI.SUM)
        # u_global_H10 = mesh.comm.allreduce(u_local_H10, op=MPI.SUM)

        # u_sobolev = np.sqrt(u_global) + k**(-2)*np.sqrt(u_global_H10)
        # u_sobolev = u_global# + k**(-2)*u_global_H10
        # return np.sqrt(error_sobolev/u_sobolev), mean_p
        norm = np.linalg.norm(reference-solution)/np.linalg.norm(reference)
        return norm
        
        #return norm, mean_p

    def conv_iterator(self,methods):
        """doc"""
        pass

    def fem_conv(self):
        """doc"""
        results_list = []
        for i in self.par_space:
            funcs.reset(ne=i)
            funcs.RBmodel.n_sens = self.n_sens
            funcs.snapshotsHandling()
            
            #funcs.getFullOrderPrior()
                    
            mc_list = []
            for m in range(self.number_mc_samples):
                funcs.calcROMprior()
                funcs.getMCpriorAndROMerror()
                funcs.getAdvancedROMPosterior()
                #funcs.getEasyROMPosterior()
                
                sample_norm = self.calc_norm(self.high_res_ref_fenics,np.copy(funcs.u_mean_y))
                mc_list.append(sample_norm)
                print("SAMPLE NORM")
                print(m)
                print(sample_norm)
            res = np.mean(mc_list)   
            print("MEAN MC") 
            results_list.append(res) 
            with open("../Results/conv_data_fem_Adv_ROM_12_08_2022_12h51.dat", "a") as file:
                file.write(str(res)+" at ne = "+str(i)+"\n")

    def statFEM_conv(self):
        """doc"""
        self.ne = 110
        funcs.reset(ne=self.ne)
        funcs.snapshotsHandling()
        funcs.calcROMprior()
        
        results_list = []
        for i in self.par_space:
            funcs.RBmodel.n_sens = i
            mc_list = []
            for m in range(self.number_mc_samples):
                self.get_noisy_data_from_solution(i,funcs.u_mean_std)
                funcs.getMCpriorAndROMerror()
                funcs.getAdvancedROMPosterior()
                sample_norm = self.calc_norm(self.high_res_ref_fenics,np.copy(funcs.u_mean_y))
                mc_list.append(sample_norm) 
                print("SAMPLE NORM")
                print(m)
                print(sample_norm)
            res = np.mean(mc_list)   
            print("MEAN MC") 
            results_list.append(res) 
            with open("../Results/conv_data_Statfem_Adv_ROM_05_12_2022_18h37.dat", "a") as file:
                file.write(str(res)+" at n_sens = "+str(i)+"\n")



    def rom_conv(self):
        """doc"""
       # self.ne = 100
       # funcs.reset(ne=self.ne)
       # funcs.RBmodel.n_sens = self.n_sens
        
        def romConv():
            self.par_space = np.arange(4,22)#22
            results_list = []
            self.sigd_list = []
            self.sigd_rom_list = []
            funcs.generateParameterSamples(256)
            for i in self.par_space:
                funcs.L = i
                #funcs.snapshotsHandling()
                #funcs.getAORAbasis(Nr=i)
                funcs.switchMesh_self("ground_truth")
                funcs.RBmodel.doFEM(freq=funcs.par1,rhsPar=np.pi**2/50,mat_par=np.array([0,0,0]))
                
                funcs.getFullOrderPrior(multiple_bases = True)
                funcs.get_noisy_data_from_solution(0,np.real(funcs.u_mean_std))
                funcs.switchMesh_self("coarse")
                
                for j,sample in enumerate(funcs.f_samples):
                    funcs.computeROMbasisSample(sample,j)
                funcs.getFullOrderPrior(multiple_bases = True)
                funcs.calcROMprior(multiple_bases = True)
                mc_list_Rom_error = []
                mc_list = []
                mc_list_easy = []
                mc_list_pred = []
                mc_sig_l =[]
                mc_sig_rom_l = []
                for m in range(self.number_mc_samples):
                    #  self.get_noisy_data_from_solution(self.n_sens,funcs.u_mean_std)
                    funcs.switchMesh_self("ground_truth")
                    funcs.RBmodel.doFEM(freq=funcs.par1,rhsPar=np.pi**2/50,mat_par=np.array([0,0,0]))
                    funcs.getFullOrderPrior(multiple_bases = True)
                    funcs.get_noisy_data_from_solution(0,np.real(funcs.u_mean_std))
                    # funcs.getMCpriorAndROMerror()

                    funcs.switchMesh_self("coarse")
                    funcs.getFullOrderPrior(multiple_bases = True)
                    funcs.calcROMprior(multiple_bases = True)
                    
                    funcs.getFullOrderPosterior()
                    funcs.getEasyROMPosterior()
                    funcs.getAdvancedROMPosterior()
                    
                    sample_norm_Rom_error = self.computeErrorNorm(solution=np.copy(funcs.dr_est),reference=funcs.dr_ex)
                
                    mc_list_Rom_error.append(sample_norm_Rom_error)
                    
                    mc_sig_l.append(funcs.RBmodel.sigd)
                    mc_sig_rom_l.append(funcs.RBmodel.sigdROM)
                self.sigd_list.append(np.mean(mc_sig_l))
                self.sigd_rom_list.append(np.mean(mc_sig_rom_l))
                resRomError = np.real(np.mean(mc_list_Rom_error)) 
                res = np.real(np.mean(mc_list)) 
                resCov = np.real(np.var(mc_list)) 
                resEasy = np.real(np.mean(mc_list_easy)) 
                resCovEasy = np.real(np.var(mc_list_easy)) 
                resPred = np.real(np.mean(mc_list_pred)) 
                resCovPred = np.real(np.var(mc_list_pred)) 
                print("MEAN MC") 
                results_list.append(res) 
                filename = "./Results/conv_data_ROM_Adv_ROM.dat"
                with open(filename, "a") as file:
                    file.write(str(res)+" at L = "+str(i)+"\n")
                with open(filename+"_COV", "a") as file:
                    file.write(str(resCov)+" at L = "+str(i)+"\n")

                filename = "./Results/conv_data_ROM_Easy_ROM.dat"
                with open(filename, "a") as file:
                    file.write(str(resEasy)+" at L = "+str(i)+"\n")
                with open(filename+"_COV", "a") as file:
                    file.write(str(resCovEasy)+" at L = "+str(i)+"\n")

                filename = "./Results/conv_data_ROM_Pred_ROM.dat"
                with open(filename, "a") as file:
                    file.write(str(resPred)+" at L = "+str(i)+"\n")
                with open(filename+"_COV", "a") as file:
                    file.write(str(resCovPred)+" at L = "+str(i)+"\n")

                filename = "./Results/conv_data_ROM_error.dat"
                with open(filename, "a") as file:
                    file.write(str(resRomError)+" at L = "+str(i)+"\n")

            filename = "./Results/sigd_data_fom.dat"
            with open(filename, "a") as file:
                for item in self.sigd_list:
                    file.write(str(item)+" at L = "+str(i)+"\n")
            filename = "./Results/sigd_data_rom.dat"
            with open(filename, "a") as file:
                for item in self.sigd_rom_list:
                    file.write(str(item)+" at L = "+str(i)+"\n")

        romConv()


        def freqError():
            self.par_space = np.arange(100,450,25)
            funcs.switchMesh_self("ground_truth")
            funcs.RBmodel.doFEM(freq=funcs.par1,rhsPar=np.pi**2/50,mat_par=np.array([0,0,0]))
            funcs.getFullOrderPrior(multiple_bases = False)
            funcs.get_noisy_data_from_solution(0,np.real(funcs.u_mean_std))
            funcs.switchMesh_self("coarse")
            funcs.generateParameterSamples(128)
            funcs.L = 15
            for j,sample in enumerate(funcs.f_samples):
                funcs.computeROMbasisSample(sample,j)
            results_list = []
            for i in self.par_space:
                funcs.par1 = i             
                #funcs.calcROMprior(multiple_bases = True)
                funcs.switchMesh_self("ground_truth")
                funcs.RBmodel.doFEM(freq=funcs.par1,rhsPar=np.pi**2/50,mat_par=np.array([0,0,0]))
                funcs.getFullOrderPrior(multiple_bases = False)
                funcs.get_noisy_data_from_solution(0,np.real(funcs.u_mean_std))
                funcs.switchMesh_self("coarse")
                #funcs.reset(ne=100)
                mc_list_OnlyRom = []
                mc_list = []
                mc_list_easy = []
                mc_list_pred = []
                mc_sig_l =[]
                mc_sig_rom_l = []
                for m in range(self.number_mc_samples):
                    funcs.switchMesh_self("ground_truth")
                    funcs.RBmodel.doFEM(freq=funcs.par1,rhsPar=np.pi**2/50,mat_par=np.array([0,0,0]))
                    funcs.getFullOrderPrior(multiple_bases = False)
                    funcs.get_noisy_data_from_solution(0,np.real(funcs.u_mean_std))
                    funcs.switchMesh_self("coarse")
                    funcs.getFullOrderPrior(multiple_bases = True)
                    funcs.calcROMprior(multiple_bases = True)
                    funcs.getFullOrderPosterior()
                    funcs.getEasyROMPosterior()
                    funcs.getAdvancedROMPosterior()

                    sample_norm_OnlyRom,_ = self.computeErrorNorm(solution=np.copy(funcs.u_mean),reference=funcs.u_mean_std)
                    sample_norm,_ = self.computeErrorNorm(solution=np.copy(funcs.u_mean_y),reference=funcs.true_process)
                    sample_normEasy,_ = self.computeErrorNorm(solution=np.copy(funcs.u_mean_y_easy),reference=funcs.true_process)
                    sample_normPred,_ = self.computeErrorNorm(solution=np.copy(funcs.u_mean_y_pred_rom),reference=funcs.true_process)
                    mc_list_OnlyRom.append(sample_norm_OnlyRom)
                    mc_list.append(sample_norm)
                    mc_list_easy.append(sample_normEasy)
                    mc_list_pred.append(sample_normPred)
                    print("SAMPLE NORM")
                    print(m)
                    print(sample_norm)
                    mc_sig_l.append(funcs.RBmodel.sigd)
                    mc_sig_rom_l.append(funcs.RBmodel.sigdROM)
                    
                    sample_norm = self.computeErrorNorm(solution=np.copy(funcs.u_mean_y),reference=funcs.true_process)
                    #sample_norm = self.calc_norm(self.high_res_ref_fenics,np.copy(funcs.u_mean))
                    mc_list.append(sample_norm)
                    print("SAMPLE NORM")
                    print(m)
                    print(sample_norm)
                resOnlyRom = np.real(np.mean(mc_list_OnlyRom)) 
                res = np.real(np.mean(mc_list)) 
                resCov = np.real(np.var(mc_list)) 
                resEasy = np.real(np.mean(mc_list_easy)) 
                resCovEasy = np.real(np.var(mc_list_easy)) 
                resPred = np.real(np.mean(mc_list_pred)) 
                resCovPred = np.real(np.var(mc_list_pred)) 
                print("MEAN MC") 
                results_list.append(res) 
                filename = "./Results/FreqError_data_ROM_Adv_ROM.dat"
                with open(filename, "a") as file:
                    file.write(str(res)+" at L = "+str(i)+"\n")
                with open(filename+"_COV", "a") as file:
                    file.write(str(resCov)+" at L = "+str(i)+"\n")

                filename = "./Results/FreqError_data_ROM_Easy_ROM.dat"
                with open(filename, "a") as file:
                    file.write(str(resEasy)+" at L = "+str(i)+"\n")
                with open(filename+"_COV", "a") as file:
                    file.write(str(resCovEasy)+" at L = "+str(i)+"\n")

                filename = "./Results/FreqError_data_ROM_Pred_ROM.dat"
                with open(filename, "a") as file:
                    file.write(str(resPred)+" at L = "+str(i)+"\n")
                with open(filename+"_COV", "a") as file:
                    file.write(str(resCovPred)+" at L = "+str(i)+"\n")

                filename = "./Results/Freq_error_ROM_OnlyRom.dat"
                with open(filename, "a") as file:
                    file.write(str(resOnlyRom)+" at L = "+str(i)+"\n")


       # freqError()


        
    # vielleicht nicht Stuff sondern plotConvergence o.Ä.?
    def plotStuff(self):
        #fig = plt.figure(figsize=(3,2))
        #plt.plot(self.high_res_ref)
        #plt.plot(self.high_res_ref_fenics)
        #plt.show()
        #fig.savefig("../Results/highres_ref.pdf", bbox_inches='tight')
        #return
        sigd_list=self.sigd_list
        sigd_rom_list = self.sigd_rom_list

        def plotROMconv():
            fig = plt.figure(figsize=(3,2))
            filenames = ["../Results/conv_data_ROM_Adv_ROM_06_12_2022_13h30_30SNAPS.dat","../Results/conv_data_ROM_Adv_ROM_06_12_2022_13h30_60SNAPS.dat","../Results/conv_data_ROM_Adv_ROM_06_12_2022_14h26_90SNAPS.dat"]
            #filename = "../Results/conv_data_ROM_Adv_ROM_06_12_2022_13h30.dat"
            snaps=0
            for filename in filenames:
                nums_rom = []
                snaps = snaps+30
                with open(filename) as file: 
                    while (line := file.readline().rstrip()):
                        num = float(line.split(' ',1)[0])
                        print(num)
                        nums_rom.append(num)
                
                plt.plot(np.arange(2,20)[0:10],nums_rom[0:10], label=str(snaps)+" Snapshots")
            plt.grid()
            plt.yscale('log')
            plt.ylabel("$L_2$ error")
            plt.xlabel("size of basis L")
            plt.legend()    
            fig.savefig("../Results/romConv.pdf", bbox_inches='tight')
        #plotROMconv()

        def plotstatROMconv():
            fig = plt.figure(figsize=(3,2))
            ax = fig.gca()
            filenames = ["../Results/conv_data_ROM_Adv_ROM_02_02_2023_16h28.dat","../Results/conv_data_ROM_Easy_ROM_02_02_2023_16h28.dat","../Results/conv_data_ROM_Pred_ROM_02_02_2023_16h28.dat"]
            #filename = "../Results/conv_data_ROM_Adv_ROM_06_12_2022_13h30.dat"
            snaps=90
            labels = ["with ROM error term", "w/o ROM error term", "predictive"]
            i=0
            plotList = []
            bases = np.arange(2,22)[0:19]
            for filename in filenames:
                nums_rom = []
                snaps = snaps+30
                with open(filename) as file: 
                    while (line := file.readline().rstrip()):
                        num = float(line.split(' ',1)[0])
                        print(num)
                        nums_rom.append(num)
                plotList.append(nums_rom)
                ax.plot(bases,nums_rom[0:19], label=labels[i])
                i+=1

            ax2 = ax.twinx()
            diff = np.divide(np.array(plotList[1])[0:19]-np.array(plotList[0])[0:19],np.array(plotList[1])[0:19])*100
            color = ['r' if d<0 else 'g' for d in diff]
            ax2.scatter(bases,diff,color=color,s=4)
            #plt.plot(np.arange(2,15)[0:13],plotList[1][0:13])
            plt.grid()
            #plt.yscale('log')
            ax.set_yscale('log')
            ax.set_ylabel("$\mathbb{E}[H_k^1]$ error")
            ax2.set_ylabel("Difference $\%$",color='red')
            ax2.tick_params(axis='y',labelcolor='red')
            plt.xlabel("size of basis L")
            #ax.legend()    
            fig.legend(bbox_transform=ax.transAxes)
            fig.savefig("../Results/statRomConvAdvVsEasyH1.pdf", bbox_inches='tight')


            fig = plt.figure(figsize=(3,2))
            plt.plot(bases,sigd_list[0:19],label="w/o ROM error term")
            plt.plot(bases,sigd_rom_list[0:19],label="with ROM error term")
            plt.yscale('log')
            plt.legend()
            plt.xlabel("size of basis L")
            plt.ylabel("$\sigma_d$")
            fig.savefig("../Results/estimatedModelError.pdf", bbox_inches='tight')
        plotstatROMconv()

        def plotFreqError():
            fig = plt.figure(figsize=(3,2))
            filenames = ["../Results/FreqError_data_ROM_Easy_ROM_08_12_2022_10h12_90SNAPS_highL.dat","../Results/FreqError_data_ROM_Pred_ROM_08_12_2022_10h12_90SNAPS_highL.dat"]
            #filename = "../Results/conv_data_ROM_Adv_ROM_06_12_2022_13h30.dat"
            snaps=90
            labels = ["statFEM", "statROM"]
            style = ["-","--"]
            i=0
            for filename in filenames:
                nums_rom = []
                snaps = snaps+30
                with open(filename) as file: 
                    while (line := file.readline().rstrip()):
                        num = float(line.split(' ',1)[0])
                        print(num)
                        nums_rom.append(num)
                
                plt.plot(np.arange(200,500,15),nums_rom, linestyle=style[i], label=labels[i])
                i+=1
            plt.grid()
            plt.yscale('log')
            plt.ylabel("$L_2$ error")
            plt.xlabel("frequency [Hz]")
            plt.legend()    
            fig.savefig("../Results/FreqErrorstatROM.pdf", bbox_inches='tight')
        #plotFreqError()
        return
        filename = "../Results/conv_data_Statfem_Adv_ROM_01_09_2022_14h23.dat"
        nums_statfem = []
        with open(filename) as file:
            while (line := file.readline().rstrip()):
                num = float(line.split(' ',1)[0])
                print(num)
                nums_statfem.append(num)
        plt.plot(nums_statfem, label="statfem_error")
        plt.yscale('log')
        plt.grid()
        plt.legend()
        plt.show()



if __name__ == '__main__':
    ConvTester = ConvergenceTester()
 #   ConvTester.compute_highres_reference(freq=funcs.par1)
 #   funcs.switchMesh_self("ground_truth")
 #   funcs.RBmodel.doFEM(freq=funcs.par1,rhsPar=np.pi**2/50,mat_par=np.array([0,0,0]))
 #   funcs.getFullOrderPrior(multiple_bases = False)
  #  
 #   funcs.get_noisy_data_from_solution(0,np.real(funcs.u_mean_std))
 #   funcs.switchMesh_self("coarse")
 #   funcs.generateParameterSamples(128)
 #   for i,sample in enumerate(funcs.f_samples):
 #       funcs.computeROMbasisSample(sample,i)
    #quit()
    #ConvTester.fem_conv()
    ConvTester.rom_conv()
   # ConvTester.plotStuff()
    #ConvTester.statFEM_conv()
