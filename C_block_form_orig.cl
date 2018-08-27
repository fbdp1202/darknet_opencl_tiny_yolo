//-------------------------------------------------------------
//
//  PROGRAM: Blocked Matrix Multipliplication kernel
//
//  PURPOSE: Computes an element of the proudct matrix
//
//              C = A * B
//
//           Using the well known blocked algorithm.  
//
//           To derive this algorithm, start with the naive
//           triply nested loop algorithm with a dot product 
//           for each element of C.  Decompose each loop 
//           into blocks of size blcksz.  This gives you 6
//           nested loops with three loops over blocks
//           and three loops over indices inside the blocks.
// 
//           Rearrange the loops to put the 3 loops over blocks 
//           at the outermost loops of the loop nest.  You'll
//           see that the three "inner" loops are just the 
//           regular matrix product between blocks.
//
//           The algorithms is simple.  Keeping all the indices
//           straight is not.  We will use the following 
//           conventions:
//
//             i,j,k            ... indices of full, global matrices 
//             Iblk, Jblk, Kblk ... indices of matrix blocks
//             iloc, jloc, kloc ... indices inside blocks
//                 
//  HISTORY: Written by Tim Mattson, November 2013 
//           Updated by Simon McIntosh-Smith, August 2014 
//
//  LICENSE: This work is licensed under the Creative Commons
//           Attribution 4.0 International License.
//           To view a copy of this license, visit
//           http://creativecommons.org/licenses/by/4.0/
//           or send a letter to:
//              Creative Commons,
//              444 Castro Street, Suite 900,
//              Mountain View, California, 94041, USA.
//
//-------------------------------------------------------------

// It turns out that the compiler generates much better code if
// we "hardwire" this block size.  16 works well for an NVIDIA 
// GPU, 32 works well for a CPU
#define rblksz 16
#define cblksz 13
#define Msize 32
#define Ksize 144
#define Nsize 169
__kernel void mmul(
                __global const float* restrict A,
                __global const float* restrict B,
                __global       float* restrict C,
                __local        float* restrict Awrk,
                __local        float* restrict Bwrk)
{
    int kloc, Kblk;
    float Ctmp=0.0f;
//32, 169
    //  This work-item will compute element C(i,j)
    const int i = get_global_id(0);
    const int j = get_global_id(1);
//(0~1), (0~12)
    // Element C(i,j) is in block C(Iblk,Jblk)
    const int Iblk = get_group_id(0);
    const int Jblk = get_group_id(1);
//16, 13
    // C(i,j) is element C(iloc, jloc) of block C(Iblk, Jblk)
    const int iloc = get_local_id(0);
    const int jloc = get_local_id(1);

    // The number of blocks are the same in each dimension
    const int Num_BLK = Ksize/rblksz; // check?

    // Setup the upper-left-corner (base address) for the A and
    // B blocks plus the increments to advance base addresses as
    // we loop over blocks
          int Abase = Iblk*Ksize*rblksz;
    const int Ainc  = rblksz;

          int Bbase = Jblk*cblksz;
    const int Binc  = rblksz*Nsize;


    // C(Iblk,Jblk) = (sum over Kblk) A(Iblk,Kblk)*B(Kblk,Jblk)
    for (Kblk = 0; Kblk < Num_BLK; Kblk++)
    {
       // Load A(Iblk,Kblk) and B(Kblk,Jblk) into local memory.
       // Each work-item loads a single element of the two blocks
       // which are shared with the entire work-group.
      if(jloc < 8){
        Awrk[(jloc*2)*rblksz+iloc] = A[Abase+(jloc*2)*Ksize+iloc];
        Awrk[(jloc*2+1)*rblksz+iloc] = A[Abase+(jloc*2+1)*Ksize+iloc];
      }

      Bwrk[iloc*cblksz+jloc] = B[Bbase+iloc*Nsize+jloc];

      barrier(CLK_LOCAL_MEM_FENCE);

       // Compute dot products over local blocks to find
       // the contribution to C(i,j) from this block
      //#pragma unroll
      for (kloc=0; kloc < rblksz; kloc++)
         Ctmp += Awrk[iloc*rblksz + kloc] * Bwrk[kloc*cblksz + jloc];

      barrier(CLK_LOCAL_MEM_FENCE);

      Abase += Ainc;
      Bbase += Binc;
    }
 
    // update global C matrix 
    C[i*Nsize+j] = Ctmp;
}


//#define BLOCK 16
__kernel void mass_mmul(
                const int M,
                const int K,
                const int N,
                __global const float* restrict A,
                __global const float* restrict B,
                __global       float* restrict C,
                __local        float* restrict Awrk,
                __local        float* restrict Bwrk)
{
    int mloc, Mblk;
    int kloc, Kblk;
    int nloc, Nblk;
    int Lblk;
//16, 16
    // C(i,j) is element C(iloc, jloc) of block C(Iblk, Jblk)
    // iglb (0 ~ 143), jglb(0 ~ 15)
    const int iglb = get_global_id(0);
    const int jglb = get_global_id(1);
    // iloc (0 ~ 15), jloc (0 ~ 15)
    const int iloc = get_local_id(0);
    const int jloc = get_local_id(1);
    // Itotal -> 144. Jtotal -> 16
    const int igrp = get_group_id(0);

    const int Itotal = get_global_size(0);
    const int Jtotal = get_global_size(1);
    // BLOCK -> 16
    const int BLOCK = get_local_size(0);

    const int ljB = jloc * BLOCK;
    //done
    const int gjKi = jglb * K + iglb;
    const int gjNi = jglb * N + iglb;

    const int giNj = iglb * N + jglb; // b position

    //done
    const int ploc = iloc + ljB;

    //done
    const int Mnum_BLK = M/Jtotal;
    const int Knum_BLK = K/Itotal;
    const int Nnum_BLK = N/Itotal;
    const int Npadding = N%Itotal;
    const int NLBase = N-Npadding;

    // local block number
    // const int Lnum_BLK = Itotal/BLOCK;

    int Mbase = 0;
    int Nbase = 0;

    const int Minc = Jtotal*K;
    const int Ninc = Itotal;

    int Abase = 0;
    int Bbase = 0;


    const int Ainc = Itotal;
    const int Binc = N*Itotal;

    int bloc;
    const int bBlk = Itotal/Jtotal;
    int Blbase;
    const int Blinc = Jtotal;
    // const int Ainc = BLOCK;
    // const int Binc = N*BLOCK;

    float Ctmp[9];
    int CMbase = 0;
    int CNbase = 0;

    const int CMinc = Jtotal*N;
    const int CNinc = Itotal;

    Mbase = 0; CMbase = 0;

    for(Mblk = 0; Mblk < Mnum_BLK; Mblk++){

      Nbase = 0; CNbase = 0;

      for(Nblk = 0; Nblk < Nnum_BLK; Nblk++){
        for(bloc = 0; bloc < bBlk; bloc++){
          Ctmp[bloc] = 0;
        }
        Abase = Mbase; Bbase = Nbase;

        for (Kblk = 0; Kblk < Knum_BLK; Kblk++){
          Awrk[ploc] = A[Abase + gjKi];
          barrier(CLK_LOCAL_MEM_FENCE);
          Blbase = Bbase;
          for(bloc = 0; bloc < bBlk; bloc++){
            Bwrk[ploc] = B[Blbase + giNj];
            barrier(CLK_LOCAL_MEM_FENCE);

            for (kloc = 0; kloc < BLOCK; kloc++)
              Ctmp[bloc] += Awrk[ljB + kloc] * Bwrk[ljB + kloc];
            barrier(CLK_LOCAL_MEM_FENCE);

            Blbase += Blinc;
          }
          Abase += Ainc;
          Bbase += Binc;
        }
        for(bloc = 0; bloc < bBlk; bloc++){
          C[CMbase + CNbase + iloc + jloc * N + bloc * BLOCK] += Ctmp[bloc];
        }

        Nbase += Ninc;
        CNbase += CNinc;

      }

      Mbase += Minc;
      CMbase += CMinc;

    }

    if(Npadding){

      Mbase = 0; CMbase = 0;

      for(Mblk = 0; Mblk < Mnum_BLK; Mblk++){
        for(bloc = 0; bloc < bBlk; bloc++){
          Ctmp[bloc] = 0;
        }
        Abase = Mbase; Bbase = NLBase; CNbase = NLBase;

        for(Kblk = 0; Kblk < Knum_BLK; Kblk++){
          Awrk[ploc] = A[Abase + gjKi];
          barrier(CLK_LOCAL_MEM_FENCE);
          Blbase = Bbase;
          for(bloc = 0; bloc < bBlk; bloc++){
            if(Npadding < iglb){
              Bwrk[ploc] = B[Blbase + giNj];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if(Npadding < iglb){
              for (kloc = 0; kloc < BLOCK; kloc++)
                Ctmp[bloc] += Awrk[ljB + kloc] * Bwrk[ljB + kloc];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            Blbase += Blinc;
          }
          Abase += Ainc;
          Bbase += Binc;

        }
        for(bloc = 0; bloc < bBlk; bloc++){
          C[CMbase + CNbase + iloc + jloc * N + bloc * BLOCK] += Ctmp[bloc];
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        Mbase += Minc;
        CMbase += CMinc;

      }
    }
}

/*
//yechan not use group, Layer total copy in global memory
#define BLOCK 16
__kernel void mass_mmul(
                const int M,
                const int K,
                const int N,
                __global const float* restrict A,
                __global const float* restrict B,
                __global       float* restrict C,
                __local        float* restrict Awrk,
                __local        float* restrict Bwrk)
{
    int mloc, Mblk;
    int kloc, Kblk;
    int nloc, Nblk;
    float Ctmp;
//16, 16
    // C(i,j) is element C(iloc, jloc) of block C(Iblk, Jblk)
    const int i = get_local_id(0);
    const int j = get_local_id(1);
    const int jB = j * BLOCK;
    const int jKi = j * K + i;
    const int jNi = j * N + i;
    const int ploc = get_local_id(0)+get_local_id(1)*BLOCK;
    const int Mnum_BLK = M/BLOCK;
    const int Knum_BLK = K/BLOCK;
    const int Nnum_BLK = N/BLOCK;
    const int Npadding = N%BLOCK;
    const int NLBase = N-Npadding;

    int Mbase = 0;
    int Nbase = 0;
    const int Minc = BLOCK*K;
    const int Ninc = BLOCK;

    int Abase = 0;
    int Bbase = 0;
    int CMbase = 0;
    int CNbase = 0;
    const int Ainc = BLOCK;
    const int Binc = N*BLOCK;
    const int CMinc = BLOCK*N;
    const int CNinc = BLOCK;

    Mbase = 0;
    CMbase = 0;
    for(Mblk = 0; Mblk < Mnum_BLK; Mblk++){
      Nbase = 0; CNbase = 0;
      for(Nblk = 0; Nblk < Nnum_BLK; Nblk++){
        Ctmp = 0.0f; Abase = Mbase; Bbase = Nbase;
        for (Kblk = 0; Kblk < Knum_BLK; Kblk++){
          Awrk[ploc] = A[Abase + jKi];
          Bwrk[ploc] = B[Bbase + jNi];
          barrier(CLK_LOCAL_MEM_FENCE);
          //#pragma unroll
          for (kloc = 0; kloc < BLOCK; kloc++)
            Ctmp += Awrk[jB + kloc] * Bwrk[kloc*BLOCK + i];
          barrier(CLK_LOCAL_MEM_FENCE);
          Abase += Ainc;
          Bbase += Binc;
        }
        C[CMbase + CNbase + jNi] = Ctmp;
        Nbase += Ninc;
        CNbase += CNinc;
      }
      Mbase += Minc;
      CMbase += CMinc;
    }

    if(Npadding){
    Mbase = 0;
    CMbase = 0;
    for(Mblk = 0; Mblk < Mnum_BLK; Mblk++){
      Ctmp = 0.0f; Abase = Mbase; Bbase = NLBase; CNbase = NLBase;
      for(Kblk = 0; Kblk < Knum_BLK; Kblk++){
        Awrk[ploc] = A[Abase + jKi];
        if(Npadding < i){
          Bwrk[ploc] = B[Bbase + jNi];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if(Npadding < i){
          for (kloc = 0; kloc < BLOCK; kloc++)
            Ctmp += Awrk[jB + kloc] * Bwrk[kloc*BLOCK + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        Abase += Ainc;
        Bbase += Binc;
      }
      C[CMbase + CNbase + jNi];
      Mbase += Minc;
      CMbase += CMinc;
    }
    }
}
*/

// hoooo
/*
__kernel void mass_mmul(
                const int param_Awid,
                const int param_Cwid,
                __global const float* restrict A,
                __global const float* restrict B,
                __global       float* restrict C,
                __local        float* restrict Awrk,
                __local        float* restrict Bwrk)
{
  
    int kloc, Kblk;
    float Ctmp;
    
//32, 169
    //  This work-item will compute element C(i,j)
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    // factor about large C -> calc Cbase
    const int e = get_global_id(2);
    const int h = param_Cwid / Nsize;
          int paramE;
          int paramH;
          int paramF;

    // factor about large A -> calc Abase
    const int g = param_Awid / Ksize;
          int paramG;

//(0~1), (0~12)
    // Element C(i,j) is in block C(Iblk,Jblk)
    const int Iblk = get_group_id(0);
    const int Jblk = get_group_id(1);
//16, 13
    // C(i,j) is element C(iloc, jloc) of block C(Iblk, Jblk)
    const int iloc = get_local_id(0);
    const int jloc = get_local_id(1);

    // The number of blocks are the same in each dimension
    const int Num_BLK = Ksize/rblksz; // small block iteration number

    // Setup the upper-left-corner (base address) for the A and
    // B blocks plus the increments to advance base addresses as
    // we loop over blocks
          int Abase;
    const int Ainc  = rblksz;

          int Bbase;
    const int Binc  = rblksz*Nsize;

          int Cbase;

    for (paramE = 0; paramE < e; paramE++){

      paramH = (int)(paramE / h);
      paramF = paramE % h;

      Cbase = paramH*param_Cwid + paramF*Nsize;

      for(paramG = 0; paramG < g; paramG++){

        Abase = (paramH*param_Awid*Msize + paramG*Ksize) + (Iblk*param_Awid*rblksz); 
          // (large block position for A) + (small block position for A)
        
        Bbase = (paramF*Ksize*Nsize*g + paramG*Ksize*Nsize) + (Jblk*cblksz);
          // (large block position for B) + (small block position for B)

        Ctmp = 0.0f;
        // C(Iblk,Jblk) = (sum over Kblk) A(Iblk,Kblk)*B(Kblk,Jblk)    
        for (Kblk = 0; Kblk < Num_BLK; Kblk++){
          // Load A(Iblk,Kblk) and B(Kblk,Jblk) into local memory.
          // Each work-item loads a single element of the two blocks
          // which are shared with the entire work-group.
          if(jloc < 8){
            Awrk[(jloc*2)*rblksz + iloc] = A[Abase + (jloc*2)*param_Awid + iloc];
            Awrk[(jloc*2+1)*rblksz + iloc] = A[Abase + (jloc*2+1)*param_Awid + iloc];
          }

          Bwrk[iloc*cblksz + jloc] = B[Bbase + iloc*Nsize + jloc];

          barrier(CLK_LOCAL_MEM_FENCE);
          // Compute dot products over local blocks to find
          // the contribution to C(i,j) from this block
          //#pragma unroll
          for (kloc = 0; kloc < rblksz; kloc++)
            Ctmp += Awrk[iloc*rblksz + kloc] * Bwrk[kloc*cblksz + jloc];

          barrier(CLK_LOCAL_MEM_FENCE);

          Abase += Ainc;
          Bbase += Binc;
        }

        C[Cbase + i*param_Cwid + j] = Ctmp;
      }
    }
    
}
*/


/*
__kernel void test_copyA(
                __global const float* restrict A,
                __global const float* restrict B,
                __global       float* restrict C,
                __local        float* restrict Awrk,
                __local        float* restrict Bwrk)
{
    int kloc, Kblk, wloc;
    float Ctmp=0.0f;
//32, 169
    //  This work-item will compute element C(i,j)
    const int i = get_global_id(0);
    const int j = get_global_id(1);
//(0~1), (0~12)
    // Element C(i,j) is in block C(Iblk,Jblk)
    const int Iblk = get_group_id(0);
    const int Jblk = get_group_id(1);
//16, 13
    // C(i,j) is element C(iloc, jloc) of block C(Iblk, Jblk)
    const int iloc = get_local_id(0);
    const int jloc = get_local_id(1);

    // The number of blocks are the same in each dimension
    const int Num_BLK = Ksize/rblksz; // check?

    // Setup the upper-left-corner (base address) for the A and
    // B blocks plus the increments to advance base addresses as
    // we loop over blocks
          int Abase = Iblk*Ksize*rblksz;
    const int Ainc  = rblksz;

          int Bbase = Jblk*cblksz;
    const int Binc  = rblksz*Nsize;

          int Cbase = Iblk*Nsize*rblksz;
    const int Cinc  = rblksz;


    // C(Iblk,Jblk) = (sum over Kblk) A(Iblk,Kblk)*B(Kblk,Jblk)
    for (Kblk = 0; Kblk < Num_BLK; Kblk++)
    {
       // Load A(Iblk,Kblk) and B(Kblk,Jblk) into local memory.
       // Each work-item loads a single element of the two blocks
       // which are shared with the entire work-group.
      if(jloc < 8){
        Awrk[(jloc*2)*rblksz+iloc] = A[Abase+(jloc*2)*Ksize+iloc];
        Awrk[(jloc*2+1)*rblksz+iloc] = A[Abase+(jloc*2+1)*Ksize+iloc];
      }

      //Bwrk[iloc*cblksz+jloc] = B[Bbase+iloc*Nsize+jloc];

      barrier(CLK_LOCAL_MEM_FENCE);

       // Compute dot products over local blocks to find
       // the contribution to C(i,j) from this block
      //#pragma unroll
      //for (kloc=0; kloc < rblksz; kloc++)
        // Ctmp += Awrk[iloc*rblksz + kloc] * Bwrk[kloc*cblksz + jloc];

      for (kloc=0; kloc < rblksz; kloc++){
        for (wloc = 0; wloc < rblksz; ++wloc)
        {
          C[Cbase + kloc*Nsize + wloc] = Awrk[kloc*rblksz + wloc];
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      Abase += Ainc;
      Bbase += Binc;
      Cbase += Cinc;
    }
 
    // update global C matrix 
//    C[i*Nsize+j] = Ctmp;
}
*/