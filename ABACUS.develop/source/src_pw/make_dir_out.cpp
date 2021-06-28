#include "../module_base/global_file.h"
#include "../module_base/global_variable.h"
#include "../module_base/global_function.h"
#ifdef __MPI
#include "mpi.h"
#endif
#include "../src_parallel/parallel_reduce.h"

//----------------------------------------------------------
// EXPLAIN : Be Called in input.cpp
//----------------------------------------------------------
void Global_File::make_dir_out(
    const std::string &suffix,
	const std::string &calculation,
    const int rank,
    const bool out_alllog)
    //const bool linear_scaling, xiaohui modify 2013-09-01. Attention! Maybe there is some problem.
    //const bool out_alllog)
{
//----------------------------------------------------------
// USE STL FUNCTION
// NAME : system
//----------------------------------------------------------

    string prefix ;

#ifdef __EPM
#ifdef __MPI
    prefix = "OUT_EPM_MPI.";
#else
    prefix = "OUT_EPM.";
#endif
#else
    prefix = "OUT.";
#endif

    global_out_dir = prefix + suffix + "/";

#ifdef __MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    int make_dir = 0;
	// mohan update 2011-05-03
	string command0 =  "test -d " + global_out_dir + " || mkdir " + global_out_dir;	

	int times = 0;
	while(times<NPROC)
	{
		if(rank==times)
		{
			if ( system( command0.c_str() ) == 0 )
			{
				cout << " MAKE THE DIR         : " << global_out_dir << endl;
				make_dir = 1;
			}
			else
			{
				cout << " PROC " << rank << " CAN NOT MAKE THE DIR !!! " << endl;	
				make_dir = 0;
			}
		}
		Parallel_Reduce::reduce_int_all(make_dir);
		if(make_dir>0)break;
		++times;
	}

#ifdef __MPI
	if(make_dir==0)
	{
		cout << " CAN NOT MAKE THE OUT DIR......." << endl;
		QUIT();		
	}
	MPI_Barrier(MPI_COMM_WORLD);
#endif

    stringstream ss,ss1;

    // mohan add 2010-09-12
    if(out_alllog)
    {
	    ss << "running_" << calculation << "_" << rank + 1;
	    open_log(ofs_running,ss.str());
    }
    else
    {
	    if(rank==0)
	    {
		    ss << "running_" << calculation;
		    open_log(ofs_running,ss.str());
	    }
    }

    if(rank==0)
    {
	    open_log(ofs_warning,"warning");
    }
    return;
}