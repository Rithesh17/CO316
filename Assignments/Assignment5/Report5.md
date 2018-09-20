# Assignment 5 - OpenMP
## By:
Anusha S (16CO102) <br>
Rithesh K (16CO253)

### All output for the code has been recorded in text files named as [program_number].txt

### Programs

1. <i>Hello World Program</i><br><br>

2. <i>Hello World Program - Version 2</i><br><br>
	The default number of threads created seems to be 56.<br><br>

3. <i>DAXPY Loop</i><br><br>
	As number of threads increases, runtime decreases. But the amount of decrease is diminishing.<br><br>

4. <i>Matrix Multiply</i><br><br>
	Similar to the above case.<br><br>

5. <i>Calculation of π</i><br><br>
	Shared variables: pi, step, num_steps<br>
	Private variable: i, x, sum(partial sum)<br><br>

6. <i>Calculation of π - Worksharing and Reduction</i><br><br>
	Given 16 numbers to add up with 16 processors in hand, the fastest way to sum up the numbers is to sum to a common, shared memory location in parallel, with each processor handling one number, with the application of appropriate constraints for memory consistency.<br><br>

7. <i>Calculation of π - Monte Carlo Simulation</i><br><br>
	Compute π by randomly choosing points, count the fraction that falls in the circle.<br><br>

8. <i>Producer-Consumer Program</i><br><br>
	Producer-consumer operation for the consumer to calculate the sum of an array after it has been populated by the producer. Additional producers and consumers can be added by simply adding a section.<br><br>

