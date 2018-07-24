# Assignment 0
## Contributors:
Rithesh K (16CO253) <br>
Anusha S (16CO102)

### Question 1:
<i>Write the device query code, compile and run it on your system. Query enough information to know all the details of
your device. Example queries: GPU card's name, GPU computation capabilities, Maximum number of block dimensions,
Maximum number of grid dimensions, Maximum size of GPU memory, Amount of constant and share memory, Warp
size, etc. Answer the following questions in your report</i><br>
1. <i>What is the architecture and compute capability of your GPU?</i><br><br>
  We are using <b>NVIDIA CUDA</b> architecture. <br><br>
2. <i>What are the maximum block dimensions for your GPU?</i><br><br>
  Maximum block dimensions: <b>[1024, 1024, 64]</b><br><br>
3. <i>Suppose you are launching a one dimensional grid and block. If the hardware's maximum grid dimension is 
65535 and the maximum block dimension is 512, what is the maximum number threads can be launched on the GPU?</i><br><br>
  Maximum block dimension: 512.<br>
  Hence the maximum number of threads in one block would be: <b>512</b>.<br><br>
4. <i>Under what conditions might a programmer choose not want to launch the maximum number of threads?</i><br><br>
  
5. <i>What can limit a program from launching the maximum number of threads on a GPU?</i><br><br>

6. <i>What is shared memory? How much shared memory is on your GPU?</i><br><br>
  Data stored in the shared memory is accessible by <b>all the threads inside a block</b>.<br>
  <b>Location: </b>Inside the GPU chip.<br>
  <b>Duration: </b>Till the block exist.<br><br>
  Shared memory in the GPU: <b>49152</b> bytes per block.

7. <i>What is global memory? How much global memory is on your GPU?</i><br><br>
  Data stored in the global memory is accessible by <b>all the threads within the application</b> (including the host).<br>
  <b>Location: </b>Outside the GPU chip.<br>
  <b>Duration: </b>Till the host allocation is completed.<br><br>
  Global memory in the GPU: <b>3405643776</b> bytes.

8. <i>What is constant memory? How much constant memory is on your GPU?</i><br><br>
  Data in the constant memory is accessible by all the threads in an application, but it is <b>read-only</b>. It is preferred over global memory for its reduced memory bandwidth.<br>
  <b>Location: </b>Outside the GPU chip.<br>
  <b>Duration: </b>Till the host allocation is completed.<br><br>
  Constant memory in the GPU: <b>65536</b> bytes.

9. <i>What does warp size signify on a GPU? What is your GPU’s warp size?</i><br><br>
  For invoking the kernel we specify the number of threads to be launched in a block, and the number of blocks to be launched. But the threads are not directly given the resources for execution. Instead, the threads in a block are further grouped into warps. <b>The warps in each block execute in SIMD fashion</b>.
  Warp size: <b>32</b> threads
  
10. <i>Is double precision supported on your GPU?</i><br><br>
  Yes.
