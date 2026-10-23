// 按实际 NEP 角向参数和输出组合编译融合内核；不按算例名称分派。
#pragma once
#include <hip/hiprtc.h>
#include <c10/util/Exception.h>
#include <sys/stat.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <string>
#include <cstdio>
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#include "nep_angular_rtc_source.h"

namespace matpl_angular_rtc {
inline void driver(hipError_t code,const char* operation) {
 TORCH_CHECK(code==hipSuccess,operation,": ",hipGetErrorString(code));
}
inline void rtc(hiprtcResult code,const char* operation) {
 TORCH_CHECK(code==HIPRTC_SUCCESS,operation,": ",hiprtcGetErrorString(code));
}
inline void ensure_directory(const std::string& path) {
 for(size_t pos=1;pos<=path.size();++pos) {
  if(pos!=path.size() && path[pos]!='/')continue;
  const auto prefix=path.substr(0,pos);if(prefix.empty())continue;
  TORCH_CHECK(::mkdir(prefix.c_str(),0700)==0 || errno==EEXIST,"cannot create angular HIPRTC cache directory ",prefix);
 }
 struct stat info{};TORCH_CHECK(::stat(path.c_str(),&info)==0 && S_ISDIR(info.st_mode),"invalid angular HIPRTC cache directory ",path);
}
inline bool regular_file(const std::string& path) {
 struct stat info{};return ::stat(path.c_str(),&info)==0 && S_ISREG(info.st_mode);
}
struct Module {hipModule_t module=nullptr;hipFunction_t function=nullptr;};
struct FileLock {
 int fd;
 explicit FileLock(const std::string& path):fd(::open(path.c_str(),O_CREAT|O_RDWR,0600)) {
  TORCH_CHECK(fd>=0,"cannot open angular HIPRTC cache lock: ",path);
  if(::flock(fd,LOCK_EX)!=0) {::close(fd);TORCH_CHECK(false,"cannot lock angular HIPRTC cache: ",path);}
 }
 ~FileLock(){::flock(fd,LOCK_UN);::close(fd);}
};
inline uint64_t hash(const std::string& value) {
 uint64_t result=UINT64_C(14695981039346656037);
 for(unsigned char c:value){result^=c;result*=UINT64_C(1099511628211);}
 return result;
}
inline Module load(const std::vector<char>& code) {
 Module m;driver(hipModuleLoadData(&m.module,code.data()),"hipModuleLoadData(angular)");
 const auto status=hipModuleGetFunction(&m.function,m.module,"nep_angular_fused");
 if(status!=hipSuccess){hipModuleUnload(m.module);driver(status,"hipModuleGetFunction(angular)");}
 return m;
}
inline Module get(int tt,int nn,int kb,int l4,int l5,unsigned outputs,int device) {
 static std::mutex mutex;
 static std::unordered_map<std::string,Module> modules;
 const std::string shape="t"+std::to_string(tt)+"-n"+std::to_string(nn)+"-k"+std::to_string(kb)+
  "-l4"+std::to_string(l4)+"-l5"+std::to_string(l5)+"-o"+std::to_string(outputs);
 const std::string process_key=std::to_string(device)+"|"+shape;
 std::lock_guard<std::mutex> guard(mutex);
 auto found=modules.find(process_key);if(found!=modules.end())return found->second;
 hipDeviceProp_t properties{};driver(hipGetDeviceProperties(&properties,device),"hipGetDeviceProperties");
 int major=0,minor=0,runtime=0,version=0;
 rtc(hiprtcVersion(&major,&minor),"hiprtcVersion");
 driver(hipRuntimeGetVersion(&runtime),"hipRuntimeGetVersion");driver(hipDriverGetVersion(&version),"hipDriverGetVersion");
 const std::string arch=properties.gcnArchName;
 const std::string material=std::string(kNepAngularRtcSource)+shape+arch+std::to_string(major)+'.'+std::to_string(minor)+
  '|'+std::to_string(runtime)+'|'+std::to_string(version)+"|c++17|-O3|-ffp-contract=off|v1";
 std::ostringstream digest;digest<<std::hex<<std::setw(16)<<std::setfill('0')<<hash(material);
 std::string directory;
 if(const char* custom=std::getenv("MATPL_NEP_ANGULAR_JIT_CACHE"))directory=custom;
 if(directory.empty()) {
  if(const char* xdg=std::getenv("XDG_CACHE_HOME"))directory=xdg;
  else if(const char* home=std::getenv("HOME"))directory=std::string(home)+"/.cache";
  else directory="/tmp";
  directory+="/matpl/nep_angular";
 }
 ensure_directory(directory);
 const std::string path=directory+"/angular-"+arch+'-'+shape+'-'+digest.str()+".hsaco";
 FileLock lock(path+".lock");
 Module module;bool hit=false;
 if(regular_file(path)) {
  try {
   std::ifstream input(path,std::ios::binary|std::ios::ate);const auto size=input.tellg();
   TORCH_CHECK(size>0,"empty angular HIPRTC cache");std::vector<char> code(static_cast<size_t>(size));
   input.seekg(0);input.read(code.data(),code.size());TORCH_CHECK(input,"cannot read angular HIPRTC cache");
   module=load(code);hit=true;
  }catch(const std::exception&) {TORCH_CHECK(::unlink(path.c_str())==0 || errno==ENOENT,"cannot remove invalid angular HIPRTC cache");}
 }
 if(!hit) {
  struct Program {hiprtcProgram p=nullptr;~Program(){if(p)hiprtcDestroyProgram(&p);}} program;
  rtc(hiprtcCreateProgram(&program.p,kNepAngularRtcSource,"nep_angular_fused.hip",0,nullptr,nullptr),"hiprtcCreateProgram");
  std::vector<std::string> options={"--std=c++17","-O3","-ffp-contract=off","--gpu-architecture="+arch,
    "-DJIT_TT="+std::to_string(tt),"-DJIT_NN="+std::to_string(nn),"-DJIT_KB="+std::to_string(kb),
    "-DJIT_L4="+std::to_string(l4),"-DJIT_L5="+std::to_string(l5),"-DJIT_OUTPUTS="+std::to_string(outputs)};
  std::vector<const char*> raw;for(const auto& option:options)raw.push_back(option.c_str());
  const auto status=hiprtcCompileProgram(program.p,raw.size(),raw.data());
  size_t log_size=0;hiprtcGetProgramLogSize(program.p,&log_size);std::string log(log_size,'\0');
  if(log_size)hiprtcGetProgramLog(program.p,log.data());
  TORCH_CHECK(status==HIPRTC_SUCCESS,"angular HIPRTC compile failed ",shape,": ",hiprtcGetErrorString(status),"\n",log);
  size_t size=0;rtc(hiprtcGetCodeSize(program.p,&size),"hiprtcGetCodeSize");std::vector<char> code(size);
  rtc(hiprtcGetCode(program.p,code.data()),"hiprtcGetCode");module=load(code);
  const std::string temporary=path+".tmp."+std::to_string(::getpid());
  try {
   std::ofstream output(temporary,std::ios::binary|std::ios::trunc);output.write(code.data(),code.size());output.close();
   TORCH_CHECK(output,"cannot write angular HIPRTC cache: ",temporary);TORCH_CHECK(::rename(temporary.c_str(),path.c_str())==0,"cannot publish angular HIPRTC cache");
  }catch(...) {::unlink(temporary.c_str());hipModuleUnload(module.module);throw;}
 }
 if(const char* log=std::getenv("MATPL_NEP_JIT_LOG"))if(log[0]=='1')
  std::fprintf(stderr,"[MatPL Angular HIPRTC] %s %s device=%d\n",hit?"cache-hit":"compiled",path.c_str(),device);
 modules.emplace(process_key,module);return module;
}
inline void launch(int atoms,int tt,int nn,int kb,int l4,int l5,unsigned outputs,int device,
 int neighbors,int feat2,int feat3,double rc,double rcinv,
 const int64_t* nl,const double* rij,const double* coeff,const int64_t* types,const double* grad,const double* sums,
 double* context,double* out,double* dc,double* ds) {
 const auto module=get(tt,nn,kb,l4,l5,outputs,device);
 void* args[]={&neighbors,&feat2,&feat3,&rc,&rcinv,&nl,&rij,&coeff,&types,&grad,&sums,&context,&out,&dc,&ds};
 const auto stream=c10::hip::getCurrentHIPStream(device);
 driver(hipModuleLaunchKernel(module.function,atoms,1,1,32,1,1,0,stream.stream(),args,nullptr),"hipModuleLaunchKernel(angular_fused)");
}
}
