from enum import Enum
from subprocess import Popen, PIPE
import os, sys
import time

class JobStatus (Enum) :
    unsubmitted = 1 #
    waiting = 2 # PD
    running = 3 # R
    terminated = 4
    finished = 5
    unknown = 100

class SlurmJob(object):
    def __init__(self, job_id=None, status=JobStatus.unsubmitted, user=None, name=None, nodes=None, nodelist=None, partition=None) -> None:
        self.job_id = job_id
        self.status = status
        self.user = user
        self.name = name
        self.partition=partition
        self.nodes = nodes
        self.nodelist = nodelist
    
    def set_cmd(self, submit_cmd):
        #such as "sbatch main_MD_test.sh"
        self.submit_cmd = submit_cmd
    
    def set_tag(self, tag):
        self.job_finish_tag = tag

    def submit(self):
        # ret = Popen([self.submit_cmd + " " + self.job_script], stdout=PIPE, stderr=PIPE, shell = True)
        ret = Popen([self.submit_cmd], stdout=PIPE, stderr=PIPE, shell = True)
        stdout, stderr = ret.communicate()
        if str(stderr, encoding='ascii') != "":
            raise RuntimeError (stderr)
        job_id = str(stdout, encoding='ascii').replace('\n','').split()[-1]
        self.job_id = job_id
        status = self.update_status()
        print ("# job {} submitted, status is {}".format(self.job_id, status))

    def update_status(self):
        self.status = self.check_status()
        return self.status

    def check_status (self):
        ret = Popen (["squeue --job " + self.job_id], shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = ret.communicate()
        if (ret.returncode != 0) :
            if str("Invalid job id specified") in str(stderr, encoding='ascii') :
                if os.path.exists (self.job_finish_tag) :
                    return JobStatus.finished
                else :
                    return JobStatus.terminated
            else :
                print("status command " + "squeue" + " fails to execute")
                print("erro info: " + str(stderr, encoding='ascii'))
                print("return code: " + str(ret.returncode))
                sys.exit ()
        status_line = str(stdout, encoding='ascii').split ('\n')[-2]
        status_word = status_line.split ()[4]
        if      status_word in ["PD","CF","S"] :
            return JobStatus.waiting
        elif    status_word in ["R","CG"] :
            return JobStatus.running
        elif    status_word in ["C","E","K","BF","CA","CD","F","NF","PR","SE","ST","TO"] :
            if os.path.exists (self.job_finish_tag) :
                return JobStatus.finished
            else :
                return JobStatus.terminated
        else :
            return JobStatus.unknown

    def running_work(self):
        self.submit()
        while True:
            status = self.check_status()
            if (status == JobStatus.waiting) or \
                (status == JobStatus.running):
                time.sleep(10)
            else:
                break
        
        assert(status == JobStatus.finished)
        return status

class Mission(object):
    def __init__(self, mission_id=None) -> None:
        self.mission_id = mission_id
        self.job_list: list[SlurmJob]= []
    
    def add_job(self, job:SlurmJob):
        self.job_list.append(job)

    def pop_job(self, job_id):
        del_job, index = self.get_job(job_id)
        self.job_list.remove(del_job)

    def get_job(self, job_id):
        for i, job in enumerate(self.job_list):
            if job.job_id == job_id:
                return job, i

    def update_job_state(self, job_id, state):
        up_job, index = self.get_job(job_id)
        up_job.status = state
        self.job_list[index] = up_job
    
    def get_running_jobs(self):
        job_list: list[SlurmJob] = []
        for job in self.job_list:
            if (job.status == JobStatus.waiting) or (job.status == JobStatus.running):
                job_list.append(job)
        return job_list

    '''
    Description: 
    job_finish_tag does not exist means this job running in error 
    param {*} self
    Returns: 
    Author: WU Xingxing
    '''
    def get_error_jobs(self):
        job_list: list[SlurmJob] = []
        for job in self.job_list:
            if os.path.exists(job.job_finish_tag) is False:
                job_list.append(job)
        return job_list
    
    def all_job_finished(self):
        return True if len(self.get_error_jobs()) == 0 else False
    
    def commit_jobs(self):
        for job in self.job_list:
            if job.status == JobStatus.unsubmitted:
                    job.submit()
    
    def check_running_job(self):
        while True:
            for job in self.job_list:
                status = job.check_status()
                self.update_job_state(job.job_id, status)
            if len(self.get_running_jobs()) == 0:
                break
            else:
                time.sleep(10)

        # error_jobs = self.get_error_jobs()
        # if len(error_jobs) > 0:
        #     error_info = "job error: {}".format([_.job_id for _ in error_jobs])
        #     raise Exception(error_info)
        return True

    '''
    Description: 
    after some jobs finished with some jobs terminated, we should try to recover these terminated jobs.
    param {*} self
    Returns: 
    Author: WU Xingxing
    '''
    def re_submmit_terminated_jobs(self):
        error_jobs = self.get_error_jobs()
        if len(error_jobs) == 0:
            return
        self.job_list.clear()
        self.job_list.extend(error_jobs)
        self.reset_job_state()
        self.commit_jobs()
        self.check_running_job()

    def reset_job_state(self):
        for job in self.job_list:
            job.status == JobStatus.unsubmitted

# if __name__ == "__main__":
#     script_path1 = "/share/home/wuxingxing/codespace/active_learning_mlff_multi_job/active_learning/scf_slurm1.job"
#     tag1 = "/share/home/wuxingxing/codespace/active_learning_mlff_multi_job/active_learning/tag1_success"
#     script_path5 = "/share/home/wuxingxing/codespace/active_learning_mlff_multi_job/active_learning/scf_slurm5.job"
#     tag5 = "/share/home/wuxingxing/codespace/active_learning_mlff_multi_job/active_learning/tag5_success"
#     mission = Mission()

#     slurm_job = SlurmJob()
#     slurm_cmd = "sbatch {}".format(script_path1)
#     slurm_job.set_tag(tag1)
#     slurm_job.set_cmd(slurm_cmd)
#     mission.add_job(slurm_job)
#     slurm_job = SlurmJob()
#     slurm_cmd = "sbatch {}".format(script_path5)
#     slurm_job.set_tag(tag5)
#     slurm_job.set_cmd(slurm_cmd)
#     mission.add_job(slurm_job)

#     mission.commit_jobs()
#     mission.check_running_job()