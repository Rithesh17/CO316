# Assignment 3
## By:
Anusha S (16CO102) <br>
Rithesh K (16CO253)

### Question 1:
<i>Convert an RGB image into a gray scale image.</i><br><br>

1. <i>How many floating operations are being performed in your color conversion kernel?</i><br><br>

2. <i>Which format would be more efficient for color conversion: a 2D matrix where each entry is an RGB value or a
3D matrix where each slice in the Z axis representes a color. I.e. is it better to have color interleaved in this
application? can you name an application where the oposite is true?</i><br><br>

3. <i>How many global memory reads are being performed by your kernel?</i><br><br>

4. <i>How many global memory writes are being performed by your kernel?</i><br><br>

5. <i>Describe what possible optimizations can be implemented to your kernel to achieve a performance speedup.</i><br><br>

<br>
### Question 2:
<i>Perform Matrix Multiplication of two large integer matrices in CUDA. Answer the following questions.</i><br><br>

1. <i>How many floating operations are being performed in the matrix multiply kernel?</i><br><br>
  Floating operations = ((One multiplication + One addition) * 512) * 512 * 512 = 268435456<br><br>

2. <i>How many global memory reads are being performed by your kernel?</i><br><br>
  Global memory reads = (512 * 2) * 512 * 512 = 268435456<br><br>

3. <i>How many global memory writes are being performed by your kernel?</i><br><br>
  Global memory writes = 512 * 512 = 262144<br><br>

4. <i>Describe what possible optimizations can be implemented to your kernel to achieve a performance speedup.</i><br><br>
  Number of global memory reads required can be reduced by utilizing shared memory concept. Less global memory accesses leads to better performance.<br><br>

<br>
### Question 3:
<i>Implement a tiled dense matrix multiplication routine using shared memory.</i><br><br>

1. <i>How many floating operations are being performed in your matrix multiply kernel? explain.</i><br><br>
  Floating operations = ((One multiplication + One addition) * 512) * 512 * 512 = 268435456<br><br>

2. <i>How many global memory reads are being performed by your kernel? explain.</i><br><br>
  Global memory reads = (2) * 512 * 512 = 524288<br><br>

3. <i>How many global memory writes are being performed by your kernel? explain.</i><br><br>
  Global memory writes = 512 * 512 = 262144<br><br>

4. <i>Describe what further optimizations can be implemented to your kernel to achieve a performance speedup.</i><br><br>

5. <i>Compare the implementation difficulty of this kernel compared to the previous MP. What difficulties did you
have with this implementation?</i><br><br>
  - Checking boundary condition for tiles<br><br>

6. <i>Suppose you have matrices with dimensions bigger than the max thread dimensions. Sketch an algorithm that would perform matrix multiplication algorithm that would perform the multiplication in this case.</i><br><br>

7. <i>Suppose you have matrices that would not fit in global memory. Sketch an algorithm that would perform matrix multiplication algorithm that would perform the multiplication out of place.</i><br><br>



  

