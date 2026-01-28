from dataclasses import asdict
import os
import sys
import logging
import getopt


class opt_values: 

    """     
        struct for passed in arguments
    """
    def __init__(self):
        self.logging_level_DUMP = 5
        self.logging_level_SUMMARY = 15
        
        self.opt_force_cpu = False
        self.opt_magic = False
        self.opt_follow_mode = False
        self.opt_recover_mode = False
        self.opt_net_cfg = 'default'
        # opt_act = 'tanh'
        self.opt_act = 'sigmoid'
        self.opt_optimizer = 'ADAM'
        self.opt_momentum = float(0)
        self.opt_regular_wd = float(0)
        self.opt_scheduler = 'NONE'
        self.opt_epochs = 100
        self.opt_lr = float(0.001)
        self.opt_gamma = float(0.99)
        self.opt_step = 100
        self.opt_batch_size = 1
        self.opt_dtype = 'float64'
        self.opt_rseed = 2021
        # session and related options
        self.opt_session_name = ''
        self.opt_session_dir = ''
        self.opt_logging_file = ''
        self.opt_tensorboard_dir = ''
        self.opt_model_dir = ''
        self.opt_model_file = ''
        self.opt_run_id = ''
        
        # end session related
        self.opt_log_level = logging.INFO
        self.opt_file_log_level = logging.DEBUG
        self.opt_journal_cycle = 1
        # wandb related
        self.opt_wandb = False
        self.opt_wandb_entity = 'moleculenn'
        self.opt_wandb_project = 'MLFF_torch'
        # end wandb related
        self.opt_init_b = False
        self.opt_save_model = True

        # scheduler specific options
        self.opt_LR_milestones = None
        self.opt_LR_patience = 0
        self.opt_LR_cooldown = 0
        self.opt_LR_total_steps = None
        self.opt_LR_max_lr = 1.
        self.opt_LR_min_lr = 0.
        self.opt_LR_T_max = None
        self.opt_autograd = True
        self.opt_dp = False
        
        # layerwised Kalman Filter options
        self.opt_nselect = 24
        self.opt_groupsize= 6
        self.opt_blocksize = 5120
        self.opt_fprefactor = 2
        
        opts,args = getopt.getopt(sys.argv[1:],
        '-h-c-m-f-R-n:-a:-z:-v:-w:-u:-e:-l:-g:-t:-b:-d:-r:-s:-o:-i:-j:',
        ['help','cpu','magic','follow','recover','net_cfg=','act=','optimizer=','momentum',
         'weight_decay=','scheduler=','epochs=','lr=','gamma=','step=',
         'batch_size=','dtype=','rseed=','session=','log_level=',
         'file_log_level=','j_cycle=','init_b','save_model',
         'milestones=','patience=','cooldown=','eps=','total_steps=',
         'max_lr=','min_lr=','T_max=',
         'wandb','wandb_entity=','wandb_project=',
         'auto_grad=', 'dmirror=', 'dp=', 'nselect=', 'groupsize=', 'blocksize=', 'fprefactor='])

        for opt_name,opt_value in opts:
            if opt_name in ('-h','--help'):
                print("")
                print("Generic parameters:")
                print("     -h, --help                  :  print help info")
                print("     -c, --cpu                   :  force training run on cpu")
                print("     -m, --magic                 :  a magic flag for your testing code")
                print("     -f, --follow                :  follow a previous trained model file")
                print("     -R, --recover               :  breakpoint training")
                print("     -n cfg, --net_cfg=cfg       :  if -f/--follow is not set, specify network cfg in parameters.py")
                print("                                    eg: -n MLFF_dmirror_cfg1")
                print("                                    if -f/--follow is set, specify the model image file name")
                print("                                    eg: '-n best1' will load model image file best1.pt from session dir")
                print("     -a act, --act=act           :  specify activation_type of MLFF_dmirror")
                print("                                    current supported: [sigmoid, softplus]")
                print("     -z name, --optimizer=name   :  specify optimizer")
                print("                                    available : SGD ASGD RPROP RMSPROP ADAG")
                print("                                                ADAD ADAM ADAMW ADAMAX LBFGS")
                print("     -v val, --momentum=val      :  specify momentum parameter for optimizer")
                print("     -w val, --weight_decay=val  :  specify weight decay regularization value")
                print("     -u name, --scheduler=name   :  specify learning rate scheduler")
                print("                                    available  : LAMBDA STEP MSTEP EXP COS PLAT OC LR_INC LR_DEC")
                print("                                    LAMBDA     : lambda scheduler")
                print("                                    STEP/MSTEP : Step/MultiStep scheduler")
                print("                                    EXP/COS    : Exponential/CosineAnnealing") 
                print("                                    PLAT/OC/LR : ReduceLROnPlateau/OneCycle")
                print("                                    LR_INC     : linearly increase to max_lr")
                print("                                    LR_DEC     : linearly decrease to min_lr")
                print("     -e epochs, --epochs=epochs  :  specify training epochs")
                print("     -l lr, --lr=lr              :  specify initial training lr")
                print("     -g gamma, --gamma=gamma     :  specify gamma of StepLR scheduler")
                print("     -t step, --step=step        :  specify step_size of StepLR scheduler")
                print("     -b size, --batch_size=size  :  specify batch size")
                print("     -d dtype, --dtype=dtype     :  specify default dtype: [float64, float32]")
                print("     -r seed, --rseed=seed       :  specify random seed used in training")
                print("     -s name, --session=name     :  specify the session name, log files, tensorboards and models")
                print("                                    will be saved to subdirectory named by this session name")
                print("     -o level, --log_level=level :  specify logging level of console")
                print("                                    available: DUMP < DEBUG < SUMMARY < [INFO] < WARNING < ERROR")
                print("                                    logging msg with level >= logging_level will be displayed")
                print("     -i L, --file_log_level=L    :  specify logging level of log file")
                print("                                    available: DUMP < [DEBUG] < SUMMARY < INFO < WARNING < ERROR")
                print("                                    logging msg with level >= logging_level will be recoreded")
                print("     -j val, --j_cycle=val       :  specify journal cycle for tensorboard and data dump")
                print("                                    0: disable journaling")
                print("                                    1: record on every epoch [default]")
                print("                                    n: record on every n epochs")
                print("")
                print("scheduler specific parameters:")
                print("     --milestones=int_list       :  milestones for MultiStep scheduler")
                print("     --patience=int_val          :  patience for ReduceLROnPlateau")
                print("     --cooldown=int_val          :  cooldown for ReduceLROnPlateau")
                print("     --total_steps=int_val       :  total_steps for OneCycle scheduler")
                print("     --max_lr=float_val          :  max learning rate for OneCycle scheduler")
                print("     --min_lr=float_val          :  min learning rate for CosineAnnealing/ReduceLROnPlateau")
                print("     --T_max=int_val             :  T_max for CosineAnnealing scheduler")
                print("")
                print("     --dmirror                   :  calculate dE/dx layer by layer explicitly")
                print("     --auto_grad                 :  calculate dE/dx by autograd func")
                print("                                    using --dmirror or --auto_grad")
                print("                                    default: --auto_grad")
                print("")
                print("     --dp                    :  use dp method(emdedding net + fitting net)")
                print("                                    using --dp=True enable dp method")
                print("                                    adding -n DeepMD_cfg (see cu/parameters.py)")
                print("                                    defalt: --dp=False (see line 90)")
                print("")
                print("wandb parameters:")
                print("     --wandb                     :  ebable wandb, sync tensorboard data to wandb")
                print("     --wandb_entity=yr_account   :  your wandb entity or account (default is: moleculenn")
                print("     --wandb_project=yr_project  :  your wandb project name (default is: MLFF_torch)")
                print("")
                print("Kalman Filter parameters:")
                print("     --nselect                   :  sample force number(default:24)")
                print("     --groupsize                 :  the number of atoms for one iteration by force(default:6)")
                print("     --blocksize                 :  max weights number for KF update(default:5120)")
                print("     --fprefactor                :  LKF force prefactor(default:2)")
                print("")
                exit()
                
            elif opt_name in ('-c','--cpu'):
                self.opt_force_cpu = True
            elif opt_name in ('-m','--magic'):
                self.opt_magic = True
            elif opt_name in ('-R','--recover'):
                self.opt_recover_mode = True
                #print(opt_recover_mode)
                # opt_follow_epoch = int(opt_value)
            elif opt_name in ('-f','--follow'):
                self.opt_follow_mode = True
                # opt_follow_epoch = int(opt_value)
            elif opt_name in ('-n','--net_cfg'):
                self.opt_net_cfg = opt_value
            elif opt_name in ('-a','--act'):
                self.opt_act = opt_value
            elif opt_name in ('-z','--optimizer'):
                self.opt_optimizer = opt_value
            elif opt_name in ('-v','--momentum'):
                self.opt_momentum = float(opt_value)
            elif opt_name in ('-w','--weight_decay'):
                self.opt_regular_wd = float(opt_value)
            elif opt_name in ('-u','--scheduler'):
                self.opt_scheduler = opt_value
            elif opt_name in ('-e','--epochs'):
                self.opt_epochs = int(opt_value)
            elif opt_name in ('-l','--lr'):
                self.opt_lr = float(opt_value)
            elif opt_name in ('-g','--gamma'):
                self.opt_gamma = float(opt_value)
            elif opt_name in ('-t','--step'):
                self.opt_step = int(opt_value)
            elif opt_name in ('-b','--batch_size'):
                self.opt_batch_size = int(opt_value)
            elif opt_name in ('-d','--dtype'):
                self.opt_dtype = opt_value
            elif opt_name in ('-r','--rseed'):
                self.opt_rseed = int(opt_value)
            elif opt_name in ('-s','--session'):
                self.opt_session_name = opt_value
                self.opt_session_dir = './'+self.opt_session_name+'/'
                self.opt_logging_file = self.opt_session_dir+'train.log'
                self.opt_model_dir = self.opt_session_dir+'model/'

                #tensorboard_base_dir = self.opt_session_dir+'tensorboard/'
                if not os.path.exists(self.opt_session_dir):
                    os.makedirs(self.opt_session_dir) 
                if not os.path.exists(self.opt_model_dir):
                    os.makedirs(self.opt_model_dir)
                """
                for i in range(1000):
                    self.opt_run_id = 'run'+str(i)
                    self.opt_tensorboard_dir = tensorboard_base_dir+self.opt_run_id
                    if (not os.path.exists(self.opt_tensorboard_dir)):
                        os.makedirs(self.opt_tensorboard_dir)
                        break
                else:
                    self.opt_tensorboard_dir = ''
                    raise RuntimeError("reaches 1000 run dirs in %s, clean it" %(self.opt_tensorboard_dir))
                """

            elif opt_name in ('-o','--log_level'):
                if (opt_value == 'DUMP'):
                    self.opt_log_level = logging_level_DUMP
                elif (opt_value == 'SUMMARY'):
                    self.opt_log_level = logging_level_SUMMARY
                else:
                    self.opt_log_level = 'logging.'+opt_value
                    self.opt_log_level = eval(self.opt_log_level)
            elif opt_name in ('-i','--file_log_level'):
                if (opt_value == 'DUMP'):
                    self.opt_file_log_level = logging_level_DUMP
                elif (opt_value == 'SUMMARY'):
                    self.opt_file_log_level = logging_level_SUMMARY
                else:
                    self.opt_file_log_level = 'logging.'+opt_value
                    self.opt_file_log_level = eval(self.opt_file_log_level)
            elif opt_name in ('-j','--j_cycle'):
                self.opt_journal_cycle = int(opt_value)
            elif opt_name in ('--milestones'):
                self.opt_LR_milestones = list(map(int, opt_value.split(',')))
            elif opt_name in ('--patience'):
                self.opt_LR_patience = int(opt_value)
            elif opt_name in ('--cooldown'):
                self.opt_LR_cooldown = int(opt_value)
            elif opt_name in ('--total_steps'):
                self.opt_LR_total_steps = int(opt_value)
            elif opt_name in ('--max_lr'):
                self.opt_LR_max_lr = float(opt_value)
            elif opt_name in ('--min_lr'):
                self.opt_LR_min_lr = float(opt_value)
            elif opt_name in ('--T_max'):
                self.opt_LR_T_max = int(opt_value)
            elif opt_name in ('--wandb'):
                self.opt_wandb = True
                import wandb
            elif opt_name in ('--wandb_entity'):
                self.opt_wandb_entity = opt_value
            elif opt_name in ('--wandb_project'):
                self.opt_wandb_project = opt_value
            elif opt_name in ('--init_b'):
                self.opt_init_b = True
            elif opt_name in ('--save_model'):
                self.opt_save_model = True
            elif opt_name in ('--dmirror'):
                self.opt_autograd = False
            elif opt_name in ('--auto_grad'):
                self.opt_autograd = True
            elif opt_name in ('--dp='):
                self.opt_dp = eval(opt_value)
            elif opt_name in ('--nselect='):
                self.opt_nselect = eval(opt_value)
            elif opt_name in ('--groupsize='):
                self.opt_groupsize = eval(opt_value)
            elif opt_name in ('--blocksize='):
                self.opt_blocksize = eval(opt_value)
            elif opt_name in ('--fprefactor='):
                self.opt_fprefactor = eval(opt_value)

        logging.addLevelName(self.logging_level_DUMP, 'DUMP')
        logging.addLevelName(self.logging_level_SUMMARY, 'SUMMARY')
        self.logger = logging.getLogger('train')
        self.logger.setLevel(self.logging_level_DUMP)

        formatter = logging.Formatter("\33[0m\33[34;49m[%(name)s]\33[0m.\33[33;49m[%(levelname)s]\33[0m: %(message)s")
        handler1 = logging.StreamHandler()
        handler1.setLevel(self.opt_log_level)
        handler1.setFormatter(formatter)
        self.logger.addHandler(handler1) 

    def dump(self,msg, *args, **kwargs):
        self.logger.log(logging_level_DUMP, msg, *args, **kwargs)

    def debug(self,msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def summary(self,msg, *args, **kwargs):
        self.logger.log(logging_level_SUMMARY, msg, *args, **kwargs)

    def info(self,msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self,msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self,msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs, exc_info=True)
