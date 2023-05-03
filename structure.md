# SOC Solver Project Structure

package com.nmsolver:

- ScsData: Input data, including matrix A, P, vector b and c, corresponding to ScsData in include/scs.h in C code.

- ScsResiduals: Stores residual data, corresponding to ScsResiduals in include/scs_work.h in C code.

- ScsSettings: Saves various parameter settings as global variables, corresponding to ScsSettings and include/glbopts.h in include/scs.h in C code.

- ScsSolution: Results.

- ScsSolver: Main external interface, corresponding to the scs function in /src/scs.c.

  ```java
  /* JAVA */
  public static ScsSolution scs(
          final ScsData d,
          final ScsCone k,
          final ScsSettings stgs);
  
  /* C: this just calls scs_init, scs_solve, and scs_finish */
  scs_int scs(
      const ScsData *d, 
      const ScsCone *k, 
      const ScsSettings *stgs, 
      ScsSolution *sol, ScsInfo *info);
  ```

- ScsWorkspace: Implements the main algorithm logic, corresponding to all functions in /src/scs.c except for scs and SCS_WORK structure in include/scs_work.h.

package com.nmsolver.cones:

- ScsCone: Input data cones, dimensions of zero, positive and second-order cones.

- ScsConeWork: Responsible for cone projection, corresponding to cones.c in /src.

package com.nmsolver.linalg:

- DVector: Vector, implements various vector operations.

- DCSCMatrix: CSC sparse matrix, implements matrix multiplication with vector, transpose.

- ScsLinSys: Responsible for solving linear equations using indirect methods, corresponding to private.c in `/linsys/cpu/indirect` in `/linsys` in C code.