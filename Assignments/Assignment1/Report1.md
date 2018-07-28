# Assignment 1
## By:
Anusha S (16CO102) <br>
Rithesh K (16CO253)

### Question 2:
<i>Perform Matrix Addition of two large integer matrices in CUDA. Answer the following questions.</i><br><br>

1. <i>How many floating operations are being performed in the matrix addition kernel?</i><br><br>
  Floating operations = None. <br>
  Integer operations = Row calculation, Column calculation, Index calculation, and Addition each.<br><br>

2. <i>How many global memory reads are being performed by your kernel?</i><br><br>
  Global memory reads = 3 * n * n = 3 * 512 * 512 = 786432<br><br>

3. <i>How many global memory writes are being performed by your kernel?</i><br><br>
  Global memory writes = n * n = 512 * 512 = 262144.<br><br>

  

