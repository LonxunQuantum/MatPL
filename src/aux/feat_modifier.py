
import pathlib
import sys
import default_para as pm 

#codepath = str(pathlib.Path(__file__).parent.resolve())

#for default_para, data_loader_2type dfeat_sparse
#sys.path.append(codepath+'/../pre_data')

class feat_modifier:
    """
        modify feature parameters.
    """ 
    
    def __init__(self) -> None:
        
        return 
    """
        feat 1
    """ 
    # pass in a whole array 
    def set_feat1_numOf2bfeat(self,val):    
        pm.Ftype1_para["numOf2bfeat"] = val 
    
    def set_feat1_Rc(self,val):
        pm.Ftype1_para["Rc"] = val

    def set_feat1_Rmin(self,val):
        pm.Ftype1_para["Rm"] = val
    
    def set_feat1_iflag_grid(self,val):
        # 1, 2, 3
        pm.Ftype1_para["iflag_grid"] = val
    
    def set_feat1_fact_base(self,val):
        pm.Ftype1_para["fact_base"] = val

    def set_feat1_dR1(self,val):
        pm.Ftype1_para["dR1"] = val 

    def set_feat1_iflag_ftype(self,val):
        if val == 3 and pm.Ftype1_para["iflag_grid"][0]!=3:
            raise Exception("iflag_grid must be 3 when iflag_ftype=3") 
        pm.Ftype1_para = val 
    
    """
        feat 2
    """
    
    def set_feat2_numOf3bfeat1(self,val):
        pm.Ftype2_para["numOf3bfeat1"] = val 

    def set_feat2_numOf3bfeat2(self,val):
        pm.Ftype2_para["numOf3bfeat2"] = val 

    def set_feat2_Rc(self,val):
        pm.Ftype2_para["Rc"] = val

    def set_feat2_Rc2(self,val):
        pm.Ftype2_para["Rc2"] = val

    def set_feat2_Rmin(self,val):
        pm.Ftype2_para["Rm"] = val
    
    def set_feat2_iflag_grid(self,val):
        # 1, 2, 3
        pm.Ftype2_para["iflag_grid"] = val
    
    def set_feat2_fact_base(self,val):
        pm.Ftype2_para["fact_base"] = val

    def set_feat2_dR1(self,val):
        pm.Ftype2_para["dR1"] = val 

    def set_feat2_dR2(self,val):
        pm.Ftype2_para["dR2"] = val 
    
    def set_feat2_iflag_ftype(self,val):
        if val == 3 and pm.Ftype1_para["iflag_grid"][0]!=3:
            raise Exception("iflag_grid must be 3 when iflag_ftype=3") 
        pm.Ftype2_para = val 
    
    """
        feat 3
    """

    def set_feat3_Rc(self,val):
        pm.Ftype3_para["Rc"] = val 
    
    def set_feat3_n2b(self,val):
        pm.Ftype3_para["n2b"] = val 

    def set_feat3_w(self,val):
        pm.Ftype3_para["2"] = val 

    """
        feat 4 
    """ 

    def set_feat4_Rc(self,val):
        pm.Ftype4_para["Rc"] = val 
    
    def set_feat4_n3b(self,val):
        pm.Ftype4_para["n3b"] = val 

    def set_feat4_zeta(self,val):
        pm.Ftype4_para["zeta"] = val 
    
    def set_feat4_w(self,val):
        pm.Ftype4_para["w"] = val 
    

    """
        feat 7: dp-chebyshev
    """
    def set_feat7_Rc(self,val):
        
        pm.Ftype7_para["Rc"] = val 
    
    def set_feat7_Rc2(self,val):
        pm.Ftype7_para["Rc2"] = val

    def set_feat7_Rm(self,val):
        pm.Ftype7_para["Rm"] = val
    
    def set_feat7_M(self,val):
        pm.Ftype7_para["M"] = val 
    
    def set_feat7_M2(self,val):
        pm.Ftype7_para["M2"] = val 
    
    def set_feat7_weight_r(self,val):
        pm.Ftype7_para["weight_r"] = val 


    


        

    