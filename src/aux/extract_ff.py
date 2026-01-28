"""
    Formulating FF file  
    WLJ, 2023
    
    WARNING: for NN ONLY SUPPORTS feature 1 & 2 at this moment.  
    
"""
import numpy as np 

table_feature_name= {   1:'gen_2b_feature', 2:'gen_3b_feature',
                        3:'gen_2bgauss_feature', 4:'gen_3bcos_feature',
                        5:'gen_MTP_feature', 6:'gen_SNAP_feature',
                        7:'gen_deepMD1_feature', 8:'gen_deepMD2_feature'} 

def write_featinfo(outfile):
    """
        0                 ! icflag 
        2                 ! num of featt
        1
        2
        3  100
        3  207  207
        73  207  207
        8  207  207
    """

    fin = open("fread_dfeat/feat.info","r")
    raw = fin.readlines()
    fin.close()
    
    # get feature indx used 
    num_feature_type = int(raw[1])
    feature_type = [] 
    
    for i in range(2,2+num_feature_type):
        feature_type.append(int(raw[i])) 
    
    num_atom_type = int(raw[2+num_feature_type].split()[0].strip())
    
    effective_lines = num_atom_type + num_feature_type + 3
    
    for idx,line in enumerate(raw):
        if idx >= effective_lines:
            break 
        if "," in line:
            outfile.writelines(line.replace(","," ").strip() + '\n')
        else:
            outfile.writelines(line.strip() + '\n')

    return feature_type, num_atom_type

def write_feat_calc_para(outfile,feat_type):
    """
        input/gen_feature_name.in 
        Write in the order given by feat_types
    """
    # grid_type = {}
    
    for feat_idx in feat_type:
        name = table_feature_name[feat_idx]+".in"
        fin = open("input/"+name,"r")
        raw = fin.readlines() 
        fin.close()
        
        """
        if feat_idx == 1:
            grid_type[1] = raw[3].split("!")[0].split(",")[2]
            # print (raw[3].split("!")[0].split(",")) 
        elif feat_idx == 2:
            grid_type[2] = raw[3].split("!")[0].split(",")[3]
        """

        for idx,line in enumerate(raw):
            outfile.writelines(line.replace(","," ").strip().split("!")[0]+"\n")
    
    # return grid_type
    
def write_feat_grid(outfile,num_atom_type):
    """
        grid info for feature calculation 
        Write in the order given by feat_types
    """
    name_list = []
    for grid_idx in range(1, num_atom_type+1):
        name_list.append("output/grid2b_type3."+str(grid_idx)) 

    for grid_idx in range(1, num_atom_type+1):
        name_list.append("output/grid3b_b1b2_type3."+str(grid_idx))

    for grid_idx in range(1, num_atom_type+1):
        name_list.append("output/grid3b_cb12_type3."+str(grid_idx))

    for name in name_list:
        fin = open(name,"r")
        raw = fin.readlines()
        fin.close() 

        for line in raw:
            outfile.writelines(line)
    """
    for feat_idx, feat in enumerate(feat_type):
        # only for 1 & 2
        if feat != 1 and feat != 2:
            continue
        
        name_list = []

        # feature idx, grid idx, grid sub-idx    
        head = [] 
        
        if feat == 1:  
            prefix = "output/grid2b_type" + grid_idx[feat]
            
            for type_idx, atom_type in enumerate(range(1,num_atom_type+1)):
                name_list.append(prefix+"."+str(atom_type))
                head.append([feat,grid_idx[feat],0,atom_type])

        if feat ==2:
            prefix = "output/grid3b_b1b2_type" + grid_idx[feat]
            
            for type_idx, atom_type in enumerate(range(1,num_atom_type+1)):
                name_list.append(prefix+"."+str(atom_type))
                head.append([feat,grid_idx[feat],0,atom_type])
                
            prefix = "output/grid3b_cb12_type" + grid_idx[feat]
            
            for type_idx, atom_type in enumerate(range(1,num_atom_type+1)):
                name_list.append(prefix+"."+str(atom_type))
                head.append([feat,grid_idx[feat],1,atom_type])

        for head_num, name in zip(head,name_list):
            fin = open(name,"r")
            raw = fin.readlines()
            fin.close() 
            
            outfile.writelines(str(head_num[0])+" "+str(head_num[1])+" "+str(head_num[2])+" "+str(head_num[3])+"\n")

            for line in raw:
                outfile.writelines(line)
    """

def extract_ff( ff_name = "myforcefield.ff",
                model_type = 3, 
                atom_type = None, 
                max_neigh_num = 100, 
                is_fitting_recon = False, 
                is_embedding_recon = False, 
                is_type_embedding = False):
    """
        We need the following:
        	i. Network params/ 
			ii. input/gen_feature_name.in 
			iii. output/grid_info 
            iv. fread_dfeat/feat.info 
    """ 
    with open(ff_name,"w") as outfile:

        if model_type == 1:
            # type of the model
            outfile.writelines(str(model_type)+"\n")
            outfile.writelines("\n")

            # feature type list
            feature_type, num_atom_type= write_featinfo(outfile)
            outfile.writelines("\n")

            # feature calculation paras
            # grid_idx = write_feat_calc_para(outfile,feature_type)
            write_feat_calc_para(outfile,feature_type)
            outfile.writelines("\n")
            
            # feature grid paras 
            # only for 1 & 2! 
            if 1 in feature_type or 2 in feature_type:
            # write_feat_grid(outfile, feature_type, num_atom_type, grid_idx)
                write_feat_grid(outfile, num_atom_type)
                outfile.writelines("\n")

            # The feature is then shifted and scaled
            for idx in range(1, len(atom_type)+1):
                ff_name = "fread_dfeat/feat_shift."+str(idx)
                fin = open(ff_name,"r")
                raw = fin.readlines()
                fin.close() 
                
                for line in raw:
                    outfile.writelines(line.strip() + "\n")
            # Weight for feature
            """
            input/weight_feat.* 
            Write in the order given by atom_type
            
            for idx in range(1, len(atom_type)+1):
                name = "fread_dfeat/weight_feat."+str(idx)
                fin = open(name,"r")
                raw = fin.readlines()
                fin.close() 
                
                for line in raw:
                    outfile.writelines(line.strip() + "\n")

            # outfile.writelines("\n")
            """
            # model_coefficients
            model_coefficients_name = "fread_dfeat/linear_fitB.ntype"
            fin = open(model_coefficients_name,"r")
            raw = fin.readlines()
            fin.close()

            for line in raw:
                outfile.writelines(line.strip() + "\n")

            # reading vdw part
            vdw_name = "fread_dfeat/vdw_fitB.ntype"
            fin = open(vdw_name,"r")
            raw = fin.readlines()
            fin.close()

            for line in raw:
                outfile.writelines(line.strip() + "\n")
            
            print("force field file saved")
        
        if model_type == 3:
            # type of the model
            outfile.writelines(str(model_type)+"\n")
            outfile.writelines("\n")

            # feature type list
            feature_type, num_atom_type= write_featinfo(outfile)
            outfile.writelines("\n")

            # feature calculation paras
            # grid_idx = write_feat_calc_para(outfile,feature_type)
            write_feat_calc_para(outfile,feature_type)
            outfile.writelines("\n")
            
            # feature grid paras 
            # only for 1 & 2! 
            if 1 in feature_type or 2 in feature_type:
                # write_feat_grid(outfile, feature_type, num_atom_type, grid_idx)
                write_feat_grid(outfile, num_atom_type)
                outfile.writelines("\n")
            
            # network paramters 
            # nn parameters
            name_list = ["fread_dfeat/Wij.txt", "fread_dfeat/data_scaler.txt"] 

            for ff_name in name_list:
                fin = open(ff_name,"r")
                raw = fin.readlines()
                fin.close()     

                max_node_num = 0 
                # max node num. 
                if ff_name == "fread_dfeat/Wij.txt":
                    for line in raw:
                        
                        line_tmp = line.split()

                        if line_tmp[0] == "m12=":
                            nums = line_tmp[1].split(",")
                            
                            max_node_num = max(max_node_num,int(nums[0]))
                            max_node_num = max(max_node_num,int(nums[1]))       
                    # print("max node num:", max_node_num)
                
                    outfile.writelines(str(max_node_num)+"\n")

                for line in raw:
                    
                    outfile.writelines(line)  

            # reading vdw part
            vdw_name = "fread_dfeat/vdw_fitB.ntype"
            fin = open(vdw_name,"r")
            raw = fin.readlines()
            fin.close()

            for line in raw:
                outfile.writelines(line.strip() + "\n")

            print("force field file saved")

        if model_type == 5: 
            # dp parameters
            # write model type
            outfile.writelines(str(model_type)+"\n")
            outfile.writelines("\n")
            
            # write reconnect info 
            if is_fitting_recon == True:
                outfile.writelines("1\n")
            else:
                outfile.writelines("0\n")

            # feat.info
            if atom_type is None:
                raise Exception("atom type list is required as input")
            
            outfile.writelines(str(len(atom_type))+" "+str(max_neigh_num)+"\n")

            for atom in atom_type:
                outfile.writelines(str(atom)+"\n")
            
            outfile.writelines("\n")

            # embedding net 
            embeding_net_name  = "embeding.net"        
            fin = open(embeding_net_name,"r")
            raw = fin.readlines()
            fin.close() 
            
            for line in raw:
                outfile.writelines(line)  
            
            outfile.writelines("\n")

            # fitting net 
            fitting_net_name  = "fitting.net"        
            fin = open(fitting_net_name,"r")
            raw = fin.readlines()
            fin.close() 
            
            for line in raw:
                outfile.writelines(line)  

            outfile.writelines("\n")

            # fitting net reconnect para
            if is_fitting_recon == True:
                recon_name = "fittingNet.resnet"
                fin = open(recon_name,"r")
                raw = fin.readlines()
                fin.close() 
                
                for line in raw:
                    outfile.writelines(line)  

                outfile.writelines("\n")

            # gen_dp.in 
            gen_dp_name  = "gen_dp.in"        
            fin = open(gen_dp_name,"r")
            raw = fin.readlines()
            fin.close() 
            
            for line in raw:
                outfile.writelines(line)
            
            outfile.writelines("\n")

            # type_physical_properties
            type_physical_properties_name = "type_physical_properties.in"
            fin = open(type_physical_properties_name,"r")
            raw = fin.readlines()
            fin.close()

            for line in raw:
                outfile.writelines(line)

            outfile.writelines("\n")
            
            print("force field file saved")
            #raise Exception("DP not supported at this moment")
            
if __name__ =="__main__":

    extract_ff(ff_name = "Li-Ta-O.ff", model_type=5)
