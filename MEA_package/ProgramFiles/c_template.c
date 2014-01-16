#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
/* Header files with a description of contents used */
#include <cvode/cvode.h>             /* prototypes for CVODE fcts., consts. */
#include <nvector/nvector_serial.h>  /* serial N_Vector types, fcts., macros */
#include <cvode/cvode_dense.h>       /* prototype for CVDense */
#include <sundials/sundials_dense.h> /* definitions DlsMat DENSE_ELEM */
#include <sundials/sundials_types.h> /* definition of type realtype */

#define Ith(v,i)    NV_Ith_S(v,i-1)       /* Ith numbers components 1..NEQ */
#define IJth(A,i,j) DENSE_ELEM(A,i-1,j-1) /* IJth numbers rows,cols 1..NEQ */

/* Problem Constants */
#define NEQ   <NEQ>                /* number of equations  */
#define RTOL  RCONST(1.0e-4)   /* scalar relative tolerance            */
#define ATOL1 RCONST(1.0e-4)   /* vector absolute tolerance components */
//#define ATOL2 RCONST(1.0e-14)
//#define ATOL3 RCONST(1.0e-6)
#define T0    RCONST(<T0>)      /* initial time           */
#define T1    RCONST(<T1>)      /* additive time      */
#define NOUT  <NOUT>               /* number of output times */
#define NPAR  <NPAR>               /* number of parameters */

//tell C++ compiler that the function is compiled in C style
extern "C"{


typedef struct {
  double parameters[NPAR];
} *UserData;


/* Functions Called by the Solver */

static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data);

static int g(realtype t, N_Vector y, realtype *gout, void *user_data);

/* Private functions to output results */

static void PrintOutput(<printoutput>);
static void PrintRootInfo(int root_f1, int root_f2);

  /* main function */
  int ftest(double* param, int nsim, double* init_sp, double* results, double* timept)
{
    char buffer[30];
    struct timeval tv;
    time_t curtime;

    time_t start,end;
    start = time(NULL);
    
    //double results[NOUT][NEQ];
    
    gettimeofday(&tv, NULL); 
    curtime=tv.tv_sec;
    
    strftime(buffer,30,"%m-%d-%Y  %T.",localtime(&curtime));

  int sim;
  int a;
  int spinit=0;

  for( sim=0; sim<nsim; sim++){  
    /*printf("particle %d\n", sim);*/
       
  realtype reltol, t, tout;
  N_Vector y, abstol;
  UserData data;
  void *cvode_mem;
  int flag, flagr, iout, sp;
  int rootsfound[2];

  y = abstol = NULL;
  cvode_mem = NULL;
  data = NULL; 

  data = (UserData) malloc(sizeof *data);
   


  y = N_VNew_Serial(NEQ);
  abstol = N_VNew_Serial(NEQ); 
  /* Set the scalar relative tolerance */
  reltol = RTOL;
  /* Set the vector absolute tolerance */
    for(sp=0; sp<NEQ; sp++){
      Ith(abstol,sp+1)=ATOL1;
    }

  /* Initialize y */
    for(sp=0; sp<NEQ; sp++){
      Ith(y,sp+1)=init_sp[sim*NEQ+sp];
    }    
    
  /* fill in the data vector */
    for (a=0; a<NPAR; a++){      
      data->parameters[a]=param[sim*NPAR+a];
   }
    
  /* Call CVodeCreate to create the solver memory and specify the 
   * Backward Differentiation Formula and the use of a Newton iteration */
  cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
  /* Define the UserData*/
  flag = CVodeSetUserData(cvode_mem, data);
  /* Call CVodeInit to initialize the integrator memory and specify the
   * user's right hand side function in y'=f(t,y), the inital time T0, and
   * the initial dependent variable vector y. */
  flag = CVodeInit(cvode_mem, f, T0, y);
  /* Call CVodeSVtolerances to specify the scalar relative tolerance
   * and vector absolute tolerances */
  flag = CVodeSVtolerances(cvode_mem, reltol, abstol);
  /* Call CVDense to specify the CVDENSE dense linear solver */
  flag = CVDense(cvode_mem, NEQ);
  /* In loop, call CVode, print results, and test for error.
     Break out of loop when NOUT preset output times have been reached.  */
  iout = 1;  tout = T1;
  //printf("spinit %d and spfin %d for initialization", spinit, spinit+NEQ-1);
  for(sp = 0; sp < NEQ; sp++){
     results[spinit+sp]=Ith(y,sp+1);
  }
  spinit = spinit+sp;
  while(1) {
    flag = CVode(cvode_mem, tout, y, &t, CV_NORMAL);
    //printf(" Sim %d, Time %f and spinit=%d \n", sim, tout, spinit);
    if (flag == CV_SUCCESS) {
      iout++;
      /*tout += T1;*/
      tout = timept[iout];
    }
    
    for(sp = 0; sp < NEQ; sp++){
      results[spinit+sp]=Ith(y,sp+1);
      }
    spinit = spinit+sp;
    if (iout == NOUT) break;
  
    }
  //PrintOutput(t, Ith(y,1), Ith(y,2), Ith(y,3), Ith(y,4), Ith(y,5), Ith(y,6), Ith(y,7), Ith(y,8), Ith(y,9));
  /* Free y and abstol vectors */
  N_VDestroy_Serial(y);
  N_VDestroy_Serial(abstol);

  /* Free integrator memory */
  CVodeFree(&cvode_mem);
  free(data);

  } 
    end = time(NULL);
    
    /*    printf("Loop used %f seconds.\n", difftime(end, start));*/

    gettimeofday(&tv, NULL); 
    curtime=tv.tv_sec;
    
    strftime(buffer,30,"%m-%d-%Y  %T.",localtime(&curtime));
    //printf("%s%ld\n",buffer,tv.tv_usec);
    
  return(0);
}


/*
 *-------------------------------
 * Functions called by the solver
 *-------------------------------
 */

/*
 * f routine. Compute function f(t,y). 
 */

static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
    
    realtype <LHS>;

  UserData data;
  data = (UserData) user_data;
<Ith>
    <eqns>
      

    
/*    
    realtype y1, y2, y3, yd1, yd3;
    
    y1 = Ith(y,1); y2 = Ith(y,2); y3 = Ith(y,3);
    
    yd1 = Ith(ydot,1) = RCONST(-0.04)*y1 + RCONST(1.0e4)*y2*y3;
    yd3 = Ith(ydot,3) = RCONST(3.0e7)*y2*y2;
    Ith(ydot,2) = -yd1 - yd3;
*/    
    return(0);

    
}

/*
 * g routine. Compute functions g_i(t,y) for i = 0,1. 

static int g(realtype t, N_Vector y, realtype *gout, void *user_data)
{
  realtype y1, y3;

  y1 = Ith(y,1); y3 = Ith(y,3);
  gout[0] = y1 - RCONST(0.0001);
  gout[1] = y3 - RCONST(0.01);

  return(0);
}
 */

/*
 * Jacobian routine. Compute J(t,y) = df/dy. *

static int Jac(long int N, realtype t,
               N_Vector y, N_Vector fy, DlsMat J, void *user_data,
               N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
  realtype y1, y2, y3;

  y1 = Ith(y,1); y2 = Ith(y,2); y3 = Ith(y,3);

  IJth(J,1,1) = RCONST(-0.04);
  IJth(J,1,2) = RCONST(1.0e4)*y3;
  IJth(J,1,3) = RCONST(1.0e4)*y2;
  IJth(J,2,1) = RCONST(0.04); 
  IJth(J,2,2) = RCONST(-1.0e4)*y3-RCONST(6.0e7)*y2;
  IJth(J,2,3) = RCONST(-1.0e4)*y2;
  IJth(J,3,2) = RCONST(6.0e7)*y2;

  return(0);
}
 */

/*
 *-------------------------------
 * Private helper functions
 *-------------------------------
 */

static void PrintOutput(<printoutput>)
{
#if defined(SUNDIALS_EXTENDED_PRECISION)
  //printf("At t = %0.4Le      y =%14.6Le  %14.6Le  %14.6Le\n", t, y1, y2, y2);
  printf("At t = %0.4Le      y =<SEP>\n", <SEP2>);
#elif defined(SUNDIALS_DOUBLE_PRECISION)
  printf("At t = %0.4le      y =<SDP>\n", <SEP2>);
#else //here
  printf("At t = %0.4e      y =<last>\n", <SEP2>);
#endif

  return;
}

static void PrintRootInfo(int root_f1, int root_f2)
{
  printf("    rootsfound[] = %3d %3d\n", root_f1, root_f2);

  return;
}

}
