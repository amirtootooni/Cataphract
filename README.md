# Cataphract
Statistical approaches to scheduling feasibility problems

# Synthetic task set generation

The typical way to measure the performance of a guarantee test is
to randomly generate a huge number of synthetic task sets and then
verify which percentage of feasible sets pass the test. However, the way
task parameters are generated may significantly affect the result and
bias the judgement about the schedulability tests.

To generate a test set we need to generate some tasks first. In our case a task set will contain N tasks. 
Each task is defined by its period (T), computation time (C) and deadline (D). In this case D = T for all tasks.
We implicitly use the utilization (U = C/T) as a description of a task, because for a given task set utilization, 
U_set = Sum of all task utilizations in the task set.

I) Generating task set utilization

U_set for a given set is a number sampled randomly from a uniform distribution of [0, 1)

II) Generating periods

Period of some task i, T_i, is a number sampled randomly from a uniform distribution of [0, 1)

III) Computation times generation

Based on the U_set we use the UUniFast algorithm to efficiently generate task sets with uniform distribution and with
O(n) complexity. Then for the N tasks in the set we'll have a U_i for each task, such that sum of all N U_i is U_set.
Using the generated U_i and T_i we calculated the C_i of the tasks in the task set.

IV) Exact Schedulability Test 

To label the generated task sets an Exact Schedulability Test for the RM scheduling policy is used.
By using response time analysis we label the task set as schedule or not. 