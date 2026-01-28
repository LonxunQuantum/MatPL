
def comm_info():
    print("\n" + "=" * 50) 
    print("          MATPL Basic Information")
    print("=" * 50) 
    print("Version: 2025.03")
    print("Compatible pwdata: >= 0.5.0")
    print("Compatible pwact: >= 0.2.0")
    print("Contact: support@pwmat.com")
    print("Citation: https://github.com/LonxunQuantum/MatPL")
    print("Manual online: http://doc.lonxun.com/MatPL/")
    print("=" * 50)  
    print("\n\n")

def matpl_help():
    commands = [
        {
            "command": "train",
            "description": "Training command for model training.",
            "example": "MatPL train input.json"
        },
        {
            "command": "test",
            "description": "Inference/testing command for model evaluation.",
            "example": "MatPL test input.json"
        },
        {
            "command": "extract_ff",
            "description": "Extract DP or NN force field ckpt files into txt format.",
            "example": "MatPL extract_ff dp_model.ckpt"
        },
        {
            "command": "infer",
            "description": "Perform energy and force inference on a single structure file using NEP or DP models.",
            "example": "MatPL infer gpumd_nep.txt atom.config pwmat/config"
        },
        {
            "command": "totxt",
            "description": "Convert NEP ckpt file into nep5.txt format for GPUMD or Lammps.",
            "example": "MatPL totxt nep_model.ckpt"
        },
        {
            "command": "compress",
            "description": "Compress DP models for faster inference by fitting the embedding net to a polynomial.",
            "example": "MatPL compress dp_model.ckpt -d 0.01 -o 3 -s cmp_dp_model"
        },
        {
            "command": "script",
            "description": "Convert DP ckpt force field files into libtorch format for LAMMPS simulations.",
            "example": "MatPL script dp_model.ckpt"
        }
    ]

    print("MatPL Command List:\n")
    for cmd in commands:
        print(f"Command: {cmd['command']}")
        print(f"  - Description: {cmd['description']}")
        print(f"  - Example:     {cmd['example']}\n")
    
    print("\nFor more detailed command instructions, please refer to the online manual:\n")
    print("http://doc.lonxun.com/MatPL/\n\n")

